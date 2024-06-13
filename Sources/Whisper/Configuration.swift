import WhisperKit
import SwiftUI
import CoreML

public extension Whisper {
	struct Configuration {
		@AppStorage("selectedModel") public var selectedModel: String = "base"
		@AppStorage("repoName") public var repoName: String = "argmaxinc/whisperkit-coreml"
		@AppStorage("selectedLanguage") public var selectedLanguage: String = "english"
		@AppStorage("enablePromptPrefill") public var enablePromptPrefill: Bool = false
		@AppStorage("enableCachePrefill") public var enableCachePrefill: Bool = false
		@AppStorage("enableSpecialCharacters") public var enableSpecialCharacters: Bool = false
		@AppStorage("enableEagerDecoding") public var enableEagerDecoding: Bool = false
		@AppStorage("temperatureStart") public var temperatureStart: Double = 0
		@AppStorage("fallbackCount") public var fallbackCount: Double = 5
		@AppStorage("compressionCheckWindow") public var compressionCheckWindow: Double = 60
		@AppStorage("sampleLength") public var sampleLength: Double = 224
		@AppStorage("silenceThreshold") public var silenceThreshold: Double = 0.2
		@AppStorage("useVAD") public var useVAD: Bool = true
		@AppStorage("tokenConfirmationsNeeded") public var tokenConfirmationsNeeded: Double = 2
		@AppStorage("chunkingStrategy") public var chunkingStrategy: ChunkingStrategy = .none
		@AppStorage("encoderComputeUnits") public var encoderComputeUnits: MLComputeUnits = .all
		@AppStorage("decoderComputeUnits") public var decoderComputeUnits: MLComputeUnits = .all
		
		public init(
			selectedModel: String = "base",
			repoName: String = "argmaxinc/whisperkit-coreml",
			selectedLanguage: String = "english",
			enablePromptPrefill: Bool = false,
			enableCachePrefill: Bool = false,
			enableSpecialCharacters: Bool = false,
			enableEagerDecoding: Bool = false,
			temperatureStart: Double = 0,
			fallbackCount: Double = 5,
			compressionCheckWindow: Double = 60,
			sampleLength: Double = 224,
			silenceThreshold: Double = 0.2,
			useVAD: Bool = true,
			tokenConfirmationsNeeded: Double = 2,
			chunkingStrategy: ChunkingStrategy = .none,
			encoderComputeUnits: MLComputeUnits = .cpuAndGPU,
			decoderComputeUnits: MLComputeUnits = .all
		) {
			self.selectedModel = selectedModel
			self.repoName = repoName
			self.selectedLanguage = selectedLanguage
			self.enablePromptPrefill = enablePromptPrefill
			self.enableCachePrefill = enableCachePrefill
			self.enableSpecialCharacters = enableSpecialCharacters
			self.enableEagerDecoding = enableEagerDecoding
			self.temperatureStart = temperatureStart
			self.fallbackCount = fallbackCount
			self.compressionCheckWindow = compressionCheckWindow
			self.sampleLength = sampleLength
			self.silenceThreshold = silenceThreshold
			self.useVAD = useVAD
			self.tokenConfirmationsNeeded = tokenConfirmationsNeeded
			self.chunkingStrategy = chunkingStrategy
			self.encoderComputeUnits = encoderComputeUnits
			self.decoderComputeUnits = decoderComputeUnits
		}
		
		static public let `default` = Configuration()
	}
}
