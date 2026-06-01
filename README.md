# EdgeAccord

EdgeAccord is a two-phase negotiation loop for fitting a keyword spotting model onto an ESP32-WROOM-32.

Phase 1 runs shape search on an untrained network. A quantization or pruning proposal becomes a concrete model shape, the compiler emits a real `.tflite` artifact and `model_data.h`, and PlatformIO reports actual RAM and flash usage from `pio run`.

Phase 2 trains the winning shape once, fine-tunes if needed, and re-verifies the final artifact against the same hardware budget.

## Scope

- Target board: `esp32dev` for ESP32-WROOM-32.
- Framework: Arduino via PlatformIO.
- Legal levers: int8 quantization, structured filter pruning, and input feature reduction.
- No int4 quantization, no multi-board support, no camera/vision, no chip generation flow.

## Layout

- `firmware/` contains the PlatformIO project and TFLite Micro entrypoint.
- `host/` contains the training, compilation, constraint, agent, and negotiation tools.
- `tests/` contains unit and integration tests.

## Environment

Set `GEMINI_API_KEY` in your environment before using the Gemini-backed agent.

Example `.env` entries:

```env
GEMINI_API_KEY=your_key_here
PIO_EXE=pio
```

## Commands

Shape search:

```bash
python -m host.negotiator --phase shape_search --use-stub-agent
```

Training pass:

```bash
python -m host.negotiator --phase train --toy-data
```

Train directly:

```bash
python -m host.train_model --toy-data
```

## Notes

- `firmware/src/model_data.h` is generated and overwritten each round.
- The memory budget comes from `host/hardware_specs.json`.
- Real on-device latency and real-world accuracy should only be documented after the ESP32 is flashed and measured.
