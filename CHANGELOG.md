# Ollama Manager - Concurrent Model Limit Feature

## What's New

Added a **Max Concurrent Models** feature that allows you to limit how many models can be loaded in memory at the same time. This helps manage memory usage and automatically unloads older models when you reach the limit.

## Key Features

### 1. Automatic Model Unloading
- When the concurrent model limit is reached, the **oldest loaded model is automatically unloaded** before loading a new one
- No more manual memory management - the system handles it for you
- Perfect for systems with limited RAM/VRAM

### 2. Default Setting: 1 Model
- The default limit is set to **1 model** at a time
- When you load a second model, the first one is automatically unloaded
- This ensures optimal memory usage on most systems

### 3. Configurable via Settings Menu
Access the new setting through the main menu:
- Press **[S]** for Settings
- Select **[6] Set Max Concurrent Models**

### 4. Flexible Options
- **1 model**: One at a time mode (recommended for low memory systems)
- **2-3 models**: Keep multiple models loaded (for systems with more RAM/VRAM)
- **0 (unlimited)**: Keep all models loaded (requires lots of memory)
- **Maximum limit**: 20 models

## How It Works

### Example with Limit = 1 (Default)
1. You load `qwen3:4b` → Model loaded successfully
2. You load `llama3.2` → System automatically unloads `qwen3:4b`, then loads `llama3.2`
3. You load `mistral` → System automatically unloads `llama3.2`, then loads `mistral`

### Example with Limit = 2
1. You load `qwen3:4b` → Model loaded successfully
2. You load `llama3.2` → Both models now loaded
3. You load `mistral` → System automatically unloads `qwen3:4b` (oldest), keeps `llama3.2`, loads `mistral`

### Example with Limit = 0 (Unlimited)
1. You load `qwen3:4b` → Model loaded successfully
2. You load `llama3.2` → Both models stay loaded
3. You load `mistral` → All three models stay loaded
4. Models stay in memory until manually unloaded or container restarts

## Files Modified

### utils.py
- Added `max_concurrent_models: int = 1` field to `OllamaConfig` dataclass
- Default value is 1 (auto-unload enabled by default)

### ollama_manager.py
- **ConfigManager.load_config()**: Now loads `MaxConcurrentModels` from config file
- **ConfigManager.save_config()**: Now saves `MaxConcurrentModels` to config file
- **handle_settings()**: 
  - Added display of current concurrent model limit
  - Added option **[6] Set Max Concurrent Models**
  - Added handler for changing the limit with detailed explanations

### docker_manager.py
- **load_model()**: 
  - Now checks if model is already loaded (prevents duplicate loads)
  - Checks current number of loaded models against the limit
  - Automatically unloads oldest model(s) when limit is reached
  - Shows helpful messages about auto-unload behavior
  - Displays current limit status after loading

## Configuration File

The setting is saved in `ollama-config.json`:

```json
{
    "UseGPU": false,
    "NetworkAccess": false,
    "OllamaPort": 11434,
    "ContainerName": "ollama-wsl",
    "MaxConcurrentModels": 1
}
```

## User Experience Improvements

### Visual Feedback
When loading a model with auto-unload enabled:
```
⚠ Concurrent model limit (1) reached
ℹ Auto-unloading 1 oldest model(s)...

  Unloading: qwen3:4b

ℹ Loading model into memory: llama3.2
  This may take a moment depending on model size...
✓ Model llama3.2 has been loaded into memory
  Model will stay loaded for 5 minutes of inactivity
  (Auto-unload is enabled - new models will replace this one)
```

### Settings Display
The settings menu now shows:
```
CONCURRENT MODEL LIMIT:
  Limit: 1 model (auto-unload enabled)
  When loading a new model, the previous one will be automatically unloaded
```

Or with multiple models:
```
CONCURRENT MODEL LIMIT:
  Limit: 3 models at once
  When loading model #4, the oldest will be unloaded
```

## Benefits

1. **Memory Efficiency**: Prevents running out of memory by limiting loaded models
2. **Automatic Management**: No need to manually unload models before loading new ones
3. **Flexibility**: Choose the limit that works best for your system
4. **User-Friendly**: Clear messages explain what's happening
5. **Safe Defaults**: Default of 1 model works well on most systems

## Backward Compatibility

- Existing config files without `MaxConcurrentModels` will default to 1
- No breaking changes to existing functionality
- All other features continue to work as before

## Usage Tips

### For Low-Memory Systems (< 16GB RAM)
- Keep limit at **1** (default)
- Models will auto-swap as you use them

### For Medium Systems (16-32GB RAM)
- Set limit to **2-3** models
- Allows quick switching between frequently used models

### For High-Memory Systems (32GB+ RAM)
- Set limit to **3-5** models or **0** (unlimited)
- Keep all your favorite models loaded

### For GPU Users
- Consider your VRAM capacity
- Large models (13B+) may only allow 1-2 at a time even with lots of RAM
- Smaller models (1B-7B) can be loaded in multiples

## Technical Notes

- The "oldest" model is determined by the order returned from `ollama ps`
- Unloading is done gracefully using the Ollama API (`keep_alive: 0`)
- If auto-unload fails, the user is notified but loading continues
- The limit is checked before each load operation
- Already-loaded models are detected to prevent duplicate loads
