import WhisperKit
import SwiftUI
import CoreML

public extension Whisper {
	struct Configuration {
		@AppStorage("selectedModel") var selectedModel: String = "distil-large-v3_594MB"
		@AppStorage("repoName") var repoName: String = "argmaxinc/whisperkit-coreml"
		@AppStorage("selectedLanguage") var selectedLanguage: String = "english"
		@AppStorage("enablePromptPrefill") var enablePromptPrefill: Bool = false
		@AppStorage("enableCachePrefill") var enableCachePrefill: Bool = false
		@AppStorage("enableSpecialCharacters") var enableSpecialCharacters: Bool = false
		@AppStorage("enableEagerDecoding") var enableEagerDecoding: Bool = false
		@AppStorage("temperatureStart") var temperatureStart: Double = 0
		@AppStorage("fallbackCount") var fallbackCount: Double = 5
		@AppStorage("compressionCheckWindow") var compressionCheckWindow: Double = 60
		@AppStorage("sampleLength") var sampleLength: Double = 224
		@AppStorage("silenceThreshold") var silenceThreshold: Double = 0.2
		@AppStorage("useVAD") var useVAD: Bool = true
		@AppStorage("tokenConfirmationsNeeded") var tokenConfirmationsNeeded: Double = 2
		@AppStorage("chunkingStrategy") var chunkingStrategy: ChunkingStrategy = .none
		@AppStorage("encoderComputeUnits") var encoderComputeUnits: MLComputeUnits = .cpuAndGPU
		@AppStorage("decoderComputeUnits") var decoderComputeUnits: MLComputeUnits = .all
		
		public init(
			selectedModel: String = "distil-large-v3_594MB",
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
