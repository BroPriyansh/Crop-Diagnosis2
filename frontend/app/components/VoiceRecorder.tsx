import React, { useState } from "react";
import { View, Text, TouchableOpacity, StyleSheet } from "react-native";
import { Audio } from "expo-av";
import { Mic, Square, Play, Pause } from "lucide-react-native";

interface VoiceRecorderProps {
  onAudioRecorded: (audioBlob: any) => void;
  isLoading?: boolean;
}

const VoiceRecorder: React.FC<VoiceRecorderProps> = ({ onAudioRecorded, isLoading }) => {
  const [recording, setRecording] = useState<Audio.Recording | null>(null);
  const [sound, setSound] = useState<Audio.Sound | null>(null);
  const [isPlaying, setIsPlaying] = useState(false);

  // Start Recording
  const startRecording = async () => {
    try {
      const { granted } = await Audio.requestPermissionsAsync();
      if (!granted) return;

      await Audio.setAudioModeAsync({
        allowsRecordingIOS: true,
        playsInSilentModeIOS: true,
      });

      const { recording } = await Audio.Recording.createAsync(
        Audio.RecordingOptionsPresets.HIGH_QUALITY
      );
      setRecording(recording);
    } catch (err) {
      console.error("Failed to start recording", err);
    }
  };

  // Stop Recording
  const stopRecording = async () => {
    if (!recording) return;
    await recording.stopAndUnloadAsync();
    const uri = recording.getURI();
    console.log("Recorded file:", uri);
    setRecording(null);

    const { sound } = await recording.createNewLoadedSoundAsync();
    setSound(sound);
    
    // Call the callback with the audio data
    onAudioRecorded({ uri, sound });
  };

  // Play / Pause
  const togglePlayback = async () => {
    if (!sound) return;

    if (isPlaying) {
      await sound.pauseAsync();
      setIsPlaying(false);
    } else {
      await sound.playAsync();
      setIsPlaying(true);
    }
  };

  return (
    <View style={styles.container}>
      <Text style={styles.title}>🎤 Voice Recorder</Text>

      {!recording ? (
        <TouchableOpacity 
          onPress={startRecording} 
          style={[styles.button, styles.startButton]}
          disabled={isLoading}
        >
          <Mic size={24} color="white" />
          <Text style={styles.buttonText}>Start Recording</Text>
        </TouchableOpacity>
      ) : (
        <TouchableOpacity 
          onPress={stopRecording} 
          style={[styles.button, styles.stopButton]}
        >
          <Square size={24} color="white" />
          <Text style={styles.buttonText}>Stop Recording</Text>
        </TouchableOpacity>
      )}

      {sound && (
        <TouchableOpacity
          onPress={togglePlayback}
          style={[styles.button, styles.playButton]}
        >
          {isPlaying ? <Pause size={24} color="white" /> : <Play size={24} color="white" />}
          <Text style={styles.buttonText}>{isPlaying ? "Pause" : "Play"}</Text>
        </TouchableOpacity>
      )}
    </View>
  );
};

const styles = StyleSheet.create({
  container: {
    padding: 24,
    alignItems: 'center',
    backgroundColor: '#fff',
    borderRadius: 12,
    margin: 16,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.1,
    shadowRadius: 4,
    elevation: 3,
  },
  title: {
    fontSize: 18,
    fontWeight: 'bold',
    marginBottom: 16,
    textAlign: 'center',
  },
  button: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    padding: 16,
    borderRadius: 12,
    marginVertical: 8,
    minWidth: 200,
  },
  startButton: {
    backgroundColor: '#4CAF50',
  },
  stopButton: {
    backgroundColor: '#f44336',
  },
  playButton: {
    backgroundColor: '#2196F3',
  },
  buttonText: {
    color: 'white',
    marginLeft: 8,
    fontSize: 16,
    fontWeight: '600',
  },
});

export default VoiceRecorder;
