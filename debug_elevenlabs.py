
try:
    from elevenlabs.client import ElevenLabs
    print("Imported ElevenLabs from elevenlabs.client")
    client = ElevenLabs(api_key="dummy")
    print("Client attributes:", dir(client))
except ImportError:
    print("Could not import ElevenLabs from elevenlabs.client")
    try:
        import elevenlabs
        print("Imported elevenlabs module")
        print("Module attributes:", dir(elevenlabs))
    except ImportError:
        print("Could not import elevenlabs module")
