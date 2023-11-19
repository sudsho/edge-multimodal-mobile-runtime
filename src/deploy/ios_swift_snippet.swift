//
//  WakePipeline.swift
//  Sample integration for edge-multimodal-mobile-runtime
//
//  Drop the two .mlpackage files (wakeword.mlpackage, speaker.mlpackage) into
//  your Xcode project and mark them as members of the target. Xcode 15 will
//  compile them into .mlmodelc bundles automatically.
//
//  Requires iOS 16 (ML Program CoreML models). Uses AVAudioEngine to pull
//  16 kHz mono, computes the log-mel spectrogram on the audio thread, then
//  runs the wake-word head every 100 ms and gates the speaker verify on it.
//

import AVFoundation
import CoreML
import Accelerate

final class WakePipeline {

    private let wake: WakeWordModel      // auto-generated from wakeword.mlpackage
    private let speaker: SpeakerModel    // auto-generated from speaker.mlpackage
    private let melQueue = DispatchQueue(label: "mel.queue", qos: .userInteractive)
    private let engine = AVAudioEngine()

    // rolling 1s buffer of log-mel frames, 98 x 40
    private var melBuf = Array(repeating: [Float](repeating: 0, count: 40), count: 98)
    private var enrollmentEmbedding: [Float]?  // 128-dim, L2 normalized
    private let cosineThreshold: Float = 0.65
    private let wakeThreshold: Float = 0.90

    var onVerifiedWake: (() -> Void)?

    init() throws {
        let cfg = MLModelConfiguration()
        cfg.computeUnits = .cpuAndNeuralEngine
        self.wake = try WakeWordModel(configuration: cfg)
        self.speaker = try SpeakerModel(configuration: cfg)
    }

    func setEnrollment(_ emb: [Float]) {
        self.enrollmentEmbedding = emb
    }

    func start() throws {
        let session = AVAudioSession.sharedInstance()
        try session.setCategory(.playAndRecord, mode: .measurement, options: [.mixWithOthers])
        try session.setActive(true)

        let input = engine.inputNode
        let format = input.outputFormat(forBus: 0)
        input.installTap(onBus: 0, bufferSize: 1600, format: format) { [weak self] buf, _ in
            self?.handle(buffer: buf)
        }
        try engine.start()
    }

    private func handle(buffer: AVAudioPCMBuffer) {
        // ... resample to 16kHz, compute mel with vDSP, shift into melBuf ...
        // omitted for brevity; see src/audio/mel.py for the reference impl

        melQueue.async { [weak self] in
            guard let self = self else { return }
            do {
                let input = try self.melToMLArray(self.melBuf)
                let out = try self.wake.prediction(mel: input)
                let (label, prob) = self.argmaxSoftmax(out.logits)
                // wakeLabel here is one of the 12 Speech Commands v2 classes
                // the head was trained on. Swap it for whichever target word
                // you re-train on for a real wake keyword.
                if label == "stop" && prob > self.wakeThreshold {
                    self.verifySpeaker()
                }
            } catch {
                NSLog("wake inference error: %@", "\(error)")
            }
        }
    }

    private func verifySpeaker() {
        guard let enroll = enrollmentEmbedding else { return }
        // build a 2s mel window from the ring buffer (200 x 40)
        do {
            let mel2s = try self.build2sMLArray()
            let embOut = try self.speaker.prediction(mel: mel2s)
            let emb = self.mlArrayToFloats(embOut.emb)
            let sim = self.cosine(emb, enroll)
            if sim >= self.cosineThreshold {
                DispatchQueue.main.async { self.onVerifiedWake?() }
            }
        } catch {
            NSLog("speaker inference error: %@", "\(error)")
        }
    }

    // helpers (mel packing, softmax, cosine) elided for brevity
    private func melToMLArray(_ mel: [[Float]]) throws -> MLMultiArray { fatalError("impl") }
    private func build2sMLArray() throws -> MLMultiArray { fatalError("impl") }
    private func mlArrayToFloats(_ arr: MLMultiArray) -> [Float] { fatalError("impl") }
    private func argmaxSoftmax(_ arr: MLMultiArray) -> (String, Float) { fatalError("impl") }
    private func cosine(_ a: [Float], _ b: [Float]) -> Float {
        var dot: Float = 0
        for i in 0..<a.count { dot += a[i] * b[i] }
        return dot
    }
}
