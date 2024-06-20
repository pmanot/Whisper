//  Copyright © 2024 Argmax, Inc. All rights reserved.

import SwiftUI
import WhisperKit
#if canImport(UIKit)
import UIKit
#elseif canImport(AppKit)
import AppKit
#endif
import AVFoundation
import CoreML
import OSLog

/// A processor for handling Whisper-related tasks, including model loading, recording, and transcription.
@MainActor
final public class Whisper: ObservableObject {
	public private(set) var whisperKit: WhisperKit?
	@Published public var state: Whisper.State = .idle
	
	private var audioRecorder = AudioRecorder()
	@Published public var audioFileURL: URL?

	@Published public var startDate: Date? = nil
	@Published public private(set) var modelState: ModelState = .unloaded
	@Published public private(set) var configuration: Configuration
	@Published public private(set) var localModels: [String] = []
	@Published public private(set) var localModelPath: String = ""
	@Published public private(set) var availableModels: [String] = []
	@Published public private(set) var availableLanguages: [String] = []
	@Published public private(set) var loadingProgressValue: Float = 0.0
	@Published public private(set) var specializationProgressRatio: Float = 0.7
	
	@Published public var isRecording: Bool = false
	@Published public var isTranscribing: Bool = false
	@Published public private(set) var confirmedSegments: [TranscriptionSegment] = []
	@Published public private(set) var unconfirmedSegments: [TranscriptionSegment] = []
	@Published public private(set) var currentText: String = ""
	@Published public private(set) var currentChunks: [Int: (chunkText: [String], fallbacks: Int)] = [:]
	
	@Published public var bufferEnergy: [Float] = []
	@Published public var bufferSeconds: Double = 0
	@Published public var totalInferenceTime: TimeInterval = 0
	@Published public var tokensPerSecond: TimeInterval = 0
	@Published public var effectiveRealTimeFactor: TimeInterval = 0
	@Published public var effectiveSpeedFactor: TimeInterval = 0
	@Published public var currentEncodingLoops: Int = 0
	@Published public var currentLag: TimeInterval = 0
	@Published public var lastConfirmedSegmentEndSeconds: Float = 0
	@Published public var requiredSegmentsForConfirmation: Int = 4
	@Published public var prevWords: [WordTiming] = []
	@Published public var lastAgreedWords: [WordTiming] = []
	@Published public var confirmedWords: [WordTiming] = []
	@Published public var confirmedText: String = ""
	@Published public var hypothesisWords: [WordTiming] = []
	@Published public var hypothesisText: String = ""
	
	@Published public var eagerResults: [TranscriptionResult?] = []
	@Published public var lastAgreedSeconds: Float = 0.0
	
	private var firstTokenTime: TimeInterval = 0
	private var pipelineStart: TimeInterval = 0
	private var currentFallbacks: Int = 0
	private var currentDecodingLoops: Int = 0
	private var lastBufferSize: Int = 0
	private var transcriptionTask: Task<Void, Never>? = nil
	private var prevResult: TranscriptionResult?
	
	// TODO: - configure custom model path
	public init(modelFolder: URL? = nil, configuration: Configuration = .default) {
		logger.info("Initializing Whisper")
		
		self.configuration = configuration
		
		let modelURL = modelFolder ?? Bundle.module.url(forResource: "Resources/openai_whisper-base", withExtension: nil)!
		loadModelFromURL(folder: modelURL)
	}
	
	/// Fetch available models from the local storage and remote repository.
	public func fetchModels() {
		availableModels = [configuration.selectedModel]
		
		if let documents = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask).first {
			let modelPath = documents.appendingPathComponent("huggingface/models/argmaxinc/whisperkit-coreml").path
			
			if FileManager.default.fileExists(atPath: modelPath) {
				localModelPath = modelPath
				do {
					let downloadedModels = try FileManager.default.contentsOfDirectory(atPath: modelPath)
					for model in downloadedModels where !localModels.contains(model) {
						localModels.append(model)
					}
				} catch {
					print("Error enumerating files at \(modelPath): \(error.localizedDescription)")
				}
			}
		}
		
		localModels = WhisperKit.formatModelFiles(localModels)
		for model in localModels {
			if !availableModels.contains(model) {
				availableModels.append(model)
			}
		}
		
		Task(priority: .high) {
			let remoteModels = try await WhisperKit.fetchAvailableModels(from: configuration.repoName)
			for model in remoteModels {
				if !availableModels.contains(model) {
					availableModels.append(model)
				}
			}
		}
	}
	
	public func loadModelFromURL(folder: URL) {
		Task(priority: .high) {
			whisperKit = try await WhisperKit(
				computeOptions: getComputeOptions(),
				verbose: true,
				logLevel: .debug,
				prewarm: false,
				load: false,
				download: false
			)
			guard let whisperKit = self.whisperKit else { return }
						
			await MainActor.run {
				self.loadingProgressValue = self.specializationProgressRatio
				self.modelState = .downloaded
			}
			
			whisperKit.modelFolder = folder
			
			await MainActor.run {
				self.loadingProgressValue = self.specializationProgressRatio
				self.modelState = .prewarming
			}
			
			let progressBarTask = Task {
				await self.updateProgressBar(targetProgress: 0.9, maxTime: 240)
			}
			
			do {
				try await whisperKit.prewarmModels()
				progressBarTask.cancel()
			} catch {
				progressBarTask.cancel()
				modelState = .unloaded
				return
			}
			
			await MainActor.run {
				self.loadingProgressValue = self.specializationProgressRatio + 0.9 * (1 - self.specializationProgressRatio)
				self.modelState = .loading
			}
			
			try await whisperKit.loadModels()
			
			await MainActor.run {
				self.availableLanguages = Constants.languages.map { $0.key }.sorted()
				self.loadingProgressValue = 1.0
				self.modelState = whisperKit.modelState
			}
		}

	}
	
	/// Load the specified model. Optionally redownloads the model if necessary.
	public func loadModel(_ model: String, redownload: Bool = false) {
		whisperKit = nil
		Task(priority: .high) {
			whisperKit = try await WhisperKit(
				computeOptions: getComputeOptions(),
				verbose: true,
				logLevel: .debug,
				prewarm: false,
				load: false,
				download: false
			)
			guard let whisperKit = whisperKit else { return }
			
			var folder: URL?
			
			if localModels.contains(model) && !redownload {
				folder = URL(fileURLWithPath: localModelPath).appendingPathComponent(model)
			} else {
				folder = try await WhisperKit.download(variant: model, from: configuration.repoName, progressCallback: { progress in
					Task { @MainActor in
						self.loadingProgressValue = Float(progress.fractionCompleted) * self.specializationProgressRatio
						self.modelState = .downloading
					}
				})
			}
			
			await MainActor.run {
				self.loadingProgressValue = self.specializationProgressRatio
				self.modelState = .downloaded
			}
			
			if let modelFolder = folder {
				whisperKit.modelFolder = modelFolder
				
				await MainActor.run {
					self.loadingProgressValue = self.specializationProgressRatio
					self.modelState = .prewarming
				}
				
				let progressBarTask = Task {
					await self.updateProgressBar(targetProgress: 0.9, maxTime: 240)
				}
				
				do {
					try await whisperKit.prewarmModels()
					progressBarTask.cancel()
				} catch {
					progressBarTask.cancel()
					if !redownload {
						loadModel(model, redownload: true)
						return
					} else {
						modelState = .unloaded
						return
					}
				}
				
				await MainActor.run {
					self.loadingProgressValue = self.specializationProgressRatio + 0.9 * (1 - self.specializationProgressRatio)
					self.modelState = .loading
				}
				
				try await whisperKit.loadModels()
				
				await MainActor.run {
					if !self.localModels.contains(model) {
						self.localModels.append(model)
					}
					
					self.availableLanguages = Constants.languages.map { $0.key }.sorted()
					self.loadingProgressValue = 1.0
					self.modelState = whisperKit.modelState
				}
			}
		}
	}
	
	/// Get compute options based on user configuration.
	private func getComputeOptions() -> ModelComputeOptions {
		return ModelComputeOptions(audioEncoderCompute: configuration.encoderComputeUnits, textDecoderCompute: configuration.decoderComputeUnits)
	}
	
	/// Update the progress bar with a target progress and maximum time.
	private func updateProgressBar(targetProgress: Float, maxTime: TimeInterval) async {
		let initialProgress = loadingProgressValue
		let decayConstant = -log(1 - targetProgress) / Float(maxTime)
		
		let startTime = Date()
		
		while true {
			let elapsedTime = Date().timeIntervalSince(startTime)
			
			let decayFactor = exp(-decayConstant * Float(elapsedTime))
			let progressIncrement = (1 - initialProgress) * (1 - decayFactor)
			let currentProgress = initialProgress + progressIncrement
			
			await MainActor.run {
				loadingProgressValue = currentProgress
			}
			
			if currentProgress >= targetProgress {
				break
			}
			
			do {
				try await Task.sleep(for: .milliseconds(100))
			} catch {
				break
			}
		}
	}
	
	/// Transcribe audio samples in eager mode.
	private func transcribeEagerMode(_ samples: [Float]) async throws -> TranscriptionResult? {
		guard let whisperKit = whisperKit else { return nil }
		
		guard whisperKit.textDecoder.supportsWordTimestamps else {
			confirmedText = "Eager mode requires word timestamps, which are not supported by the current model: \(configuration.selectedModel)."
			return nil
		}
		
		let languageCode = Constants.languages[configuration.selectedLanguage, default: Constants.defaultLanguageCode]
		let task: DecodingTask = .transcribe
		
		let options = DecodingOptions(
			verbose: true,
			task: task,
			language: languageCode,
			temperature: Float(configuration.temperatureStart),
			temperatureFallbackCount: Int(configuration.fallbackCount),
			sampleLength: Int(configuration.sampleLength),
			usePrefillPrompt: configuration.enablePromptPrefill,
			usePrefillCache: configuration.enableCachePrefill,
			skipSpecialTokens: !configuration.enableSpecialCharacters,
			withoutTimestamps: false,
			wordTimestamps: true,
			firstTokenLogProbThreshold: -1.5
		)
		
		let decodingCallback: ((TranscriptionProgress) -> Bool?) = { progress in
			DispatchQueue.main.async {
				let fallbacks = Int(progress.timings.totalDecodingFallbacks)
				if progress.text.count < self.currentText.count {
					if fallbacks == self.currentFallbacks {
					} else {
						print("Fallback occured: \(fallbacks)")
					}
				}
				self.currentText = progress.text
				self.currentFallbacks = fallbacks
				self.currentDecodingLoops += 1
			}
			let currentTokens = progress.tokens
			let checkWindow = Int(self.configuration.compressionCheckWindow)
			if currentTokens.count > checkWindow {
				let checkTokens: [Int] = currentTokens.suffix(checkWindow)
				let compressionRatio = compressionRatio(of: checkTokens)
				if compressionRatio > options.compressionRatioThreshold! {
					return false
				}
			}
			if progress.avgLogprob! < options.logProbThreshold! {
				return false
			}
			
			return nil
		}
		
		logger.info("[EagerMode] \(self.lastAgreedSeconds)-\(Double(samples.count) / 16000.0) seconds")
		
		let streamingAudio = samples
		var streamOptions = options
		streamOptions.clipTimestamps = [lastAgreedSeconds]
		let lastAgreedTokens = lastAgreedWords.flatMap { $0.tokens }
		streamOptions.prefixTokens = lastAgreedTokens
		do {
			let transcription: TranscriptionResult? = try await whisperKit.transcribe(audioArray: streamingAudio, decodeOptions: streamOptions, callback: decodingCallback).first
			await MainActor.run {
				var skipAppend = false
				if let result = transcription {
					self.hypothesisWords = result.allWords.filter { $0.start >= self.lastAgreedSeconds }
					
					if let prevResult = self.prevResult {
						self.prevWords = prevResult.allWords.filter { $0.start >= self.lastAgreedSeconds }
						let commonPrefix = findLongestCommonPrefix(self.prevWords, self.hypothesisWords)
						logger.info("[EagerMode] Prev \"\((self.prevWords.map { $0.word }).joined())\"")
						logger.info("[EagerMode] Next \"\((self.hypothesisWords.map { $0.word }).joined())\"")
						logger.info("[EagerMode] Found common prefix \"\((commonPrefix.map { $0.word }).joined())\"")
						
						if commonPrefix.count >= Int(self.configuration.tokenConfirmationsNeeded) {
							self.lastAgreedWords = commonPrefix.suffix(Int(self.configuration.tokenConfirmationsNeeded))
							self.lastAgreedSeconds = self.lastAgreedWords.first!.start
							logger.info("[EagerMode] Found new last agreed word \"\(self.lastAgreedWords.first!.word)\" at \(self.lastAgreedSeconds) seconds")
							
							self.confirmedWords.append(contentsOf: commonPrefix.prefix(commonPrefix.count - Int(self.configuration.tokenConfirmationsNeeded)))
							let currentWords = self.confirmedWords.map { $0.word }.joined()
							logger.info("[EagerMode] Current:  \(self.lastAgreedSeconds) -> \(Double(samples.count) / 16000.0) \(currentWords)")
						} else {
							logger.info("[EagerMode] Using same last agreed time \(self.lastAgreedSeconds)")
							skipAppend = true
						}
					}
					self.prevResult = result
				}
				
				if !skipAppend {
					self.eagerResults.append(transcription)
				}
			}
		} catch {
			logger.error("[EagerMode] Error: \(error)")
		}
		
		await MainActor.run {
			let finalWords = self.confirmedWords.map { $0.word }.joined()
			self.confirmedText = finalWords
			
			let lastHypothesis = self.lastAgreedWords + findLongestDifferentSuffix(self.prevWords, self.hypothesisWords)
			self.hypothesisText = lastHypothesis.map { $0.word }.joined()
		}
		
		let mergedResult = mergeTranscriptionResults(eagerResults, confirmedWords: confirmedWords)
		
		return mergedResult
	}
	
	/// Toggle recording state.
	public func toggleRecording(shouldLoop: Bool) {
		isRecording.toggle()
		
		if isRecording {
			state = .recording
			resetState()
			startRecording(shouldLoop)
		} else {
			stopRecording(shouldLoop)
		}
	}
	
	/// Reset the state of the processor.
	private func resetState() {
		transcriptionTask?.cancel()
		isRecording = false
		isTranscribing = false
		whisperKit?.audioProcessor.stopRecording()
		currentText = ""
		currentChunks = [:]
		
		pipelineStart = Double.greatestFiniteMagnitude
		firstTokenTime = Double.greatestFiniteMagnitude
		effectiveRealTimeFactor = 0
		effectiveSpeedFactor = 0
		totalInferenceTime = 0
		tokensPerSecond = 0
		currentLag = 0
		currentFallbacks = 0
		currentEncodingLoops = 0
		currentDecodingLoops = 0
		lastBufferSize = 0
		lastConfirmedSegmentEndSeconds = 0
		requiredSegmentsForConfirmation = 2
		bufferEnergy = []
		bufferSeconds = 0
		confirmedSegments = []
		unconfirmedSegments = []
		
		eagerResults = []
		prevResult = nil
		lastAgreedSeconds = 0.0
		prevWords = []
		lastAgreedWords = []
		confirmedWords = []
		confirmedText = ""
		hypothesisWords = []
		hypothesisText = ""
	}
	
	/// Start recording audio.
	public func startRecording(_ loop: Bool) {
		self.startDate = nil
		
		if let audioProcessor = whisperKit?.audioProcessor {
			audioRecorder.startRecording()
			self.audioFileURL = audioRecorder.audioFileURL

			Task(priority: .high) {
				guard await AudioProcessor.requestRecordPermission() else {
					logger.info("Microphone access was not granted.")
					return
				}
				
				try? audioProcessor.startRecordingLive { _ in
					Task { @MainActor in
						if self.startDate == nil {
							self.startDate = .now
						}
						
						self.bufferEnergy = self.whisperKit?.audioProcessor.relativeEnergy ?? []
						self.bufferSeconds = Double(self.whisperKit?.audioProcessor.audioSamples.count ?? 0) / Double(WhisperKit.sampleRate)
					}
				}
				
				await MainActor.run {
					isRecording = true
					isTranscribing = true
					self.state = .recording
				}
				
				if loop {
					realtimeLoop()
				}
			}
		} else {
			self.state = .idle
		}
	}
	
	/// Stop recording audio.
	public func stopRecording(_ loop: Bool) {
		audioRecorder.stopRecording()
		isRecording = false
		self.state = .transcribing
		stopRealtimeTranscription()
		whisperKit?.audioProcessor.stopRecording()
		
		/*
		if !loop {
			Task(priority: .high) {
				do {
					//try await transcribeCurrentBuffer()
				} catch {
					self.state = .idle
					logger.error("Error: \(error.localizedDescription)")
				}
			}
		}*/
	}
	
	
	// TODO: - Fix
	/// Real-time transcription loop.
	private func realtimeLoop() {
		transcriptionTask = Task(priority: .high) {
			while isRecording && isTranscribing {
				do {
					try await transcribeCurrentBuffer()
				} catch {
					logger.error("Error: \(error.localizedDescription)")
					break
				}
			}
		}
	}
	
	/// Stop real-time transcription.
	private func stopRealtimeTranscription() {
		transcriptionTask?.cancel()
	}
	
	/// Transcribe the current audio buffer.
	private func transcribeCurrentBuffer() async throws {
		guard let whisperKit = whisperKit else { return }
		
		let currentBuffer = whisperKit.audioProcessor.audioSamples
		
		let nextBufferSize = currentBuffer.count - lastBufferSize
		let nextBufferSeconds = Float(nextBufferSize) / Float(WhisperKit.sampleRate)
		
		guard nextBufferSeconds > 1 else {
			await MainActor.run {
				if currentText.isEmpty {
					currentText = "Waiting for speech..."
				}
			}
			try await Task.sleep(for: .milliseconds(100))
			return
		}
		
		if configuration.useVAD {
			let voiceDetected = AudioProcessor.isVoiceDetected(
				in: whisperKit.audioProcessor.relativeEnergy,
				nextBufferInSeconds: nextBufferSeconds,
				silenceThreshold: Float(configuration.silenceThreshold)
			)
			guard voiceDetected else {
				await MainActor.run {
					if currentText.isEmpty {
						currentText = "Waiting for speech..."
					}
				}
				
				try await Task.sleep(for: .milliseconds(100))
				return
			}
		}
		
		lastBufferSize = currentBuffer.count
		
		let transcription = try await transcribeEagerMode(Array(currentBuffer))
		await MainActor.run {
			currentText = ""
			guard let segments = transcription?.segments else {
				return
			}
			
			self.tokensPerSecond = transcription?.timings.tokensPerSecond ?? 0
			self.firstTokenTime = transcription?.timings.firstTokenTime ?? 0
			self.pipelineStart = transcription?.timings.pipelineStart ?? 0
			self.currentLag = transcription?.timings.decodingLoop ?? 0
			self.currentEncodingLoops += Int(transcription?.timings.totalEncodingRuns ?? 0)
			let totalAudio = Double(currentBuffer.count) / Double(WhisperKit.sampleRate)
			self.totalInferenceTime += transcription?.timings.fullPipeline ?? 0
			self.effectiveRealTimeFactor = Double(self.totalInferenceTime) / totalAudio
			self.effectiveSpeedFactor = totalAudio / Double(self.totalInferenceTime)
			
			if segments.count > self.requiredSegmentsForConfirmation {
				let numberOfSegmentsToConfirm = segments.count - self.requiredSegmentsForConfirmation
				let confirmedSegmentsArray = Array(segments.prefix(numberOfSegmentsToConfirm))
				let remainingSegments = Array(segments.suffix(self.requiredSegmentsForConfirmation))
				
				if let lastConfirmedSegment = confirmedSegmentsArray.last, lastConfirmedSegment.end > self.lastConfirmedSegmentEndSeconds {
					self.lastConfirmedSegmentEndSeconds = lastConfirmedSegment.end
					if !self.confirmedSegments.contains(confirmedSegmentsArray) {
						self.confirmedSegments.append(contentsOf: confirmedSegmentsArray)
					}
				}
				self.unconfirmedSegments = remainingSegments
			} else {
				self.unconfirmedSegments = segments
			}
			
			isTranscribing = false
			self.state = .idle
		}
	}
	
	public func transcribe() async throws -> [TranscriptionResult] {
		guard let whisperKit = whisperKit else { return [] }
		/*
		// Fetch the current audio buffer
		let currentBuffer = whisperKit.audioProcessor.audioSamples
		if currentBuffer.isEmpty {
			logger.info("No audio samples available for transcription.")
			return []
		}
		
		// Initial setup for transcription results
		var transcriptionResults = [TranscriptionResult]()
		
		// Use a loop to process the buffer in manageable chunks if necessary
		let bufferSize = currentBuffer.count
		let sampleRate = Float(WhisperKit.sampleRate)
		var startIndex = 0
		let chunkSize = Int(sampleRate * 10) // Processing 10 seconds at a time
		
		while startIndex < bufferSize {
			let endIndex = min(startIndex + chunkSize, bufferSize)
			let bufferSlice = Array(currentBuffer[startIndex..<endIndex])
			
			do {
				if let transcriptionResult = try await transcribeEagerMode(bufferSlice) {
					transcriptionResults.append(transcriptionResult)
				}
			} catch {
				logger.error("Error during transcription: \(error.localizedDescription)")
				throw error
			}
			
			startIndex = endIndex
		}
		
		// Update state and logging
		isTranscribing = false
		state = .idle
		logger.info("Completed transcription with \(transcriptionResults.count) results.")
		
		return transcriptionResults
		*/
		guard let url = audioFileURL else {
			logger.error("URL not found")
			return []
		}
		
		let languageCode = Constants.languages[configuration.selectedLanguage, default: Constants.defaultLanguageCode]
		let task: DecodingTask = .transcribe
		
		let options = DecodingOptions(
			verbose: true,
			task: task,
			language: languageCode,
			temperature: Float(configuration.temperatureStart),
			temperatureFallbackCount: Int(configuration.fallbackCount),
			sampleLength: Int(configuration.sampleLength),
			usePrefillPrompt: configuration.enablePromptPrefill,
			usePrefillCache: configuration.enableCachePrefill,
			skipSpecialTokens: !configuration.enableSpecialCharacters,
			withoutTimestamps: false,
			wordTimestamps: true,
			firstTokenLogProbThreshold: -1.5
		)

		
		return try await whisperKit.transcribe(
			audioPath: url.path(),
			decodeOptions: options
		)
	}

}

public extension Whisper {
	enum State: Hashable {
		case recording
		case transcribing
		case idle
	}
}

class AudioRecorder: NSObject, AVCaptureAudioDataOutputSampleBufferDelegate, AVCaptureFileOutputRecordingDelegate {
	private var captureSession: AVCaptureSession?
	private var audioFileOutput: AVCaptureAudioFileOutput?
	var audioFileURL: URL?
	
	override init() {
		super.init()
		setupSession()
	}
	
	private func setupSession() {
		let session = AVCaptureSession()
		session.beginConfiguration()
		
		guard let audioDevice = AVCaptureDevice.default(for: .audio) else {
			print("Audio device is unavailable")
			return
		}
		
		do {
			let audioInput = try AVCaptureDeviceInput(device: audioDevice)
			if session.canAddInput(audioInput) {
				session.addInput(audioInput)
			} else {
				print("Cannot add audio input")
				return
			}
			
			let fileOutput = AVCaptureAudioFileOutput()
			if session.canAddOutput(fileOutput) {
				session.addOutput(fileOutput)
			} else {
				print("Cannot add file output")
				return
			}
			
			self.audioFileOutput = fileOutput
		} catch {
			print("Failed to setup audio input/output: \(error)")
			return
		}
		
		session.commitConfiguration()
		self.captureSession = session
	}
	
	func startRecording() {
		captureSession?.startRunning()
		
		let tempDir = FileManager.default.temporaryDirectory
		let fileName = "whisper_\(Date().timeIntervalSince1970).m4a"
		let fileURL = tempDir.appendingPathComponent(fileName)
		self.audioFileURL = fileURL
		
		if let fileOutput = audioFileOutput, captureSession?.isRunning == true {
			try? fileOutput.startRecording(to: fileURL, outputFileType: .m4a, recordingDelegate: self)
		} else {
			print("Session is not running or output is not available")
		}
	}
	
	func stopRecording() {
		audioFileOutput?.stopRecording()
		captureSession?.stopRunning()
	}
	
	func fileOutput(_ output: AVCaptureFileOutput, didStartRecordingTo fileURL: URL, from connections: [AVCaptureConnection]) {
		print("Recording started successfully to \(fileURL).")
	}
	
	func fileOutput(_ output: AVCaptureFileOutput, didFinishRecordingTo outputFileURL: URL, from connections: [AVCaptureConnection], error: Error?) {
		if let error = error {
			print("Recording failed with error: \(error.localizedDescription)")
		} else {
			print("Recording finished successfully to \(outputFileURL).")
		}
	}
}


// MARK: - Logging

fileprivate let logger = Logger(subsystem: "com.Whisper", category: "Whisper")
