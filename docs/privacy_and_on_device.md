# Privacy and on-device processing

The pipeline is designed so that audio never leaves the device.

## data flow

- The mic tap runs on the audio thread only. No network calls are made
  on the wake or verify paths in the deploy sketches.
- VAD, mel front-end, wake CNN, and speaker embedding all run in the
  device's local inference runtime (CoreML on iOS, ONNX Runtime Mobile
  on Android). No cloud speech APIs are used.
- Speaker enrollment: the intended shape is that enrollment audio is
  captured on device, turned into an L2-normalized 128-dim embedding,
  and stored locally. The Swift and Kotlin sketches only expose a
  `setEnrollment(emb)` setter that accepts an already-computed
  embedding; the capture flow, embedding averaging, and platform
  keystore write are not implemented here.

## what the app knows vs what the server knows

| | On device | Sent to any server |
|---|---|---|
| Raw microphone PCM | yes (in RAM only) | never |
| Log-mel spectrograms | yes | never |
| Wake-word probabilities | yes | never |
| Speaker embeddings | yes (in memory) | never |
| Wake event (fact that user woke) | yes | only if the containing app chooses to log it |

## design choices worth calling out

- Silero VAD stays on device. It also runs locally.
- No wake-word "hot phrase" is transmitted for tuning. Improvements to
  the model come from opt-in Speech Commands / VoxCeleb data
  collections, not from user recordings.
- The pipeline is intended to not maintain a rolling audio buffer beyond
  the ~2 s needed for speaker verify. Older audio would be overwritten
  in the ring buffer in a real implementation.
- Enrollment embeddings in a real integration would live in a
  hardware-backed keystore (iOS Keychain, Android Keystore-backed
  EncryptedSharedPreferences); wiring that up is left to the app.

## what still needs care in production

- Some Android OEMs default to `MediaRecorder.AudioSource.MIC` which can
  route through a system-level speech recognizer. Prefer
  `VOICE_RECOGNITION` and confirm no cloud provider is hooked in.
- iOS AVAudioSession category `.playAndRecord` with `.measurement` mode
  disables system-level speech processing. Avoid switching to
  `.voiceChat`; that path is documented to send audio to Siri servers
  on some device configurations.
- If your app also sends audio to a cloud LLM after the wake event,
  that is a separate consent surface. Show it as a distinct permission
  and do not conflate it with the on-device wake.
