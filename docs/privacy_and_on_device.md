# Privacy and on-device processing

Everything in this repo is designed so that audio never leaves the device.

## data flow

- The mic tap runs on the audio thread only. No network calls anywhere on
  the wake or verify paths.
- VAD, mel front-end, wake CNN, and speaker embedding all run in the
  device's local inference runtime (CoreML on iOS, ONNX Runtime Mobile on
  Android). No cloud speech APIs are used.
- Speaker enrollment: the user says the keyword three times. The three
  L2-normalized 128-dim embeddings are averaged and stored in:
  - iOS: Keychain (default access group, not iCloud)
  - Android: EncryptedSharedPreferences with a Keystore-backed AES key
- Raw enrollment audio is discarded after embedding. There is no way to
  reconstruct a full waveform from the stored 128-dim vector.

## what the app knows vs what the server knows

| | On device | Sent to any server |
|---|---|---|
| Raw microphone PCM | yes (in RAM only) | never |
| Log-mel spectrograms | yes | never |
| Wake-word probabilities | yes | never |
| Speaker embeddings | yes (Keychain / Keystore) | never |
| Wake event (fact that user woke) | yes | only if the containing app chooses to log it |

## design choices that mattered

- Silero VAD stays on device. It also runs locally.
- No wake-word "hot phrase" is transmitted for tuning. Improvements to the
  model happen from opt-in Speech Commands / VoxCeleb data collections,
  not from user recordings.
- The pipeline does not maintain a rolling audio buffer beyond the ~2 s
  needed for speaker verify. Older audio is overwritten in the ring buffer.
- Enrollment embeddings are stored in a hardware-backed keystore. If the
  device is wiped or the user reinstalls the app, the enrollment is lost
  and they need to re-enroll.

## what still needs care in production

- Some Android OEMs default to `MediaRecorder.AudioSource.MIC` which can
  route through a system-level speech recognizer. Use
  `VOICE_RECOGNITION` and confirm no cloud provider is hooked in.
- iOS AVAudioSession category `.playAndRecord` with
  `.measurement` mode disables system-level speech processing. Do not
  switch to `.voiceChat`; that path is documented to send audio to
  Siri servers on some device configurations.
- If your app also sends audio to a cloud LLM after the wake event,
  that is a separate consent surface. Show it as a distinct permission
  and do not conflate it with the on-device wake.
