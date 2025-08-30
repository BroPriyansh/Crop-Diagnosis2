
import React, { useEffect, useState } from "react";
import {
  View,
  TextInput,
  FlatList,
  Text,
  ActivityIndicator,
  Alert,
  TouchableOpacity,
  Platform,
  StyleSheet,
  StatusBar,
  SafeAreaView,
  KeyboardAvoidingView,
  ScrollView,
} from "react-native";
import { Audio } from "expo-av";

// Dynamic URL based on platform and environment
const getBaseURL = () => {
  let url;
  if (__DEV__) {
    if (Platform.OS === 'android') {
      // Use the IP from your error message - this should work for Android emulator
      url = 'http://172.16.92.63:8001'; 
    } else if (Platform.OS === 'ios') {
      url = 'http://localhost:8001'; // iOS simulator
    } else {
      url = 'http://localhost:8001'; // Web or other platforms
    }
  } else {
    url = 'http://localhost:8001'; // Production - adjust as needed
  }
  
  console.log(' Platform:', Platform.OS);
  console.log(' Dev mode:', __DEV__);
  console.log(' Selected URL:', url);
  return url;
};

const API_URL = `${getBaseURL()}/api/ask`;
const TTS_URL = `${getBaseURL()}/api/tts`;

console.log(' Final API_URL:', API_URL);
console.log(' Final TTS_URL:', TTS_URL);

interface Message {
  text: string;
  sender: "bot" | "user";
}

export default function ChatScreen() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [question, setQuestion] = useState("");
  const [loading, setLoading] = useState(false);
  const [sound, setSound] = useState<Audio.Sound | null>(null);
  const [ttsLoading, setTtsLoading] = useState<string | null>(null);
  const [isPlaying, setIsPlaying] = useState(false);

  // Cleanup previous sound
  const unloadSound = async () => {
    if (sound) {
      await sound.unloadAsync();
      setSound(null);
    }
    setIsPlaying(false);
  };

  // Stop audio playback
  const stopAudio = async () => {
    if (sound) {
      await sound.stopAsync();
      await unloadSound();
    }
  };

  // Play audio from blob
  const playAudio = async (blob: Blob) => {
    try {
      await unloadSound();
      setIsPlaying(true);

      const base64 = await new Promise<string>((resolve, reject) => {
        const reader = new FileReader();
        reader.onload = () => {
          const result = reader.result as string;
          resolve(result.split(",")[1]);
        };
        reader.onerror = reject;
        reader.readAsDataURL(blob);
      });

      const { sound: newSound } = await Audio.Sound.createAsync({
        uri: `data:audio/mpeg;base64,${base64}`,
      });

      setSound(newSound);

      // Set up playback status listener
      newSound.setOnPlaybackStatusUpdate((status) => {
        if (status.isLoaded && status.didJustFinish) {
          setIsPlaying(false);
        }
      });

      await newSound.playAsync();
    } catch (err) {
      console.error("Audio play error:", err);
      Alert.alert("Error", "Failed to play audio");
      setIsPlaying(false);
    }
  };

  // Convert text to speech
  const handleTTS = async (text: string, messageId: string) => {
    setTtsLoading(messageId);
    try {
      const res = await fetch(TTS_URL, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text }),
      });

      if (!res.ok) throw new Error("TTS request failed");

      const blob = await res.blob();
      await playAudio(blob);
    } catch (err) {
      console.error("TTS error:", err);
      Alert.alert("Error", "Failed to convert text to speech");
    } finally {
      setTtsLoading(null);
    }
  };

  // Handle backend response - always expect JSON now
  const handleResponse = async (res: Response, userMessage?: string) => {
    if (userMessage) {
      setMessages((prev) => [...prev, { text: userMessage, sender: "user" }]);
    }

    const data = await res.json();
    if (data.error) {
      Alert.alert("Error", data.error);
      return;
    }

    setMessages((prev) => [
      ...prev,
      { text: data.answer || "No advice received", sender: "bot" },
    ]);
  };

  // Fetch initial AI advice on mount
  useEffect(() => {
    const fetchInitialAdvice = async () => {
      console.log(' Starting fetchInitialAdvice...');
      console.log(' Using API_URL:', API_URL);
      setLoading(true);
      try {
        console.log(' Making fetch request...');
        const res = await fetch(API_URL, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({}), // No TTS initially - just get text
        });

        console.log('📡 Fetch response received:', res.status, res.statusText);
        if (!res.ok) {
          console.error(' Server error:', res.status, res.statusText);
          throw new Error(`Server responded with ${res.status}: ${res.statusText}`);
        }
        
        console.log(' Response OK, processing...');
        await handleResponse(res);
        console.log(' Initial advice loaded successfully');
      } catch (err) {
        console.error(' Fetch error:', err);
        console.error(' Error details:', {
          message: err,
          name: err,
          stack: err
        });
        Alert.alert("Connection Error", `Failed to fetch initial advice: ${err}`);
      } finally {
        setLoading(false);
      }
    };

    fetchInitialAdvice();
  }, []);

  // Send user question
  const sendQuestion = async () => {
    if (!question.trim()) return;

    setLoading(true);
    try {
      const res = await fetch(API_URL, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question }),
      });

      if (!res.ok) throw new Error("Server responded with an error");
      await handleResponse(res, question);
      setQuestion("");
    } catch (err) {
      console.error(err);
      Alert.alert("Error", "Failed to get AI response.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <SafeAreaView style={styles.container}>
      <StatusBar backgroundColor="#2e7d32" barStyle="light-content" />
      
      {/* Header */}
      <View style={styles.header}>
        <View style={styles.headerContent}>
          <Text style={styles.headerTitle}>Our Crop Recommendation</Text>
          <Text style={styles.headerSubtitle}>Your intelligent farming companio</Text>
        </View>
      </View>

      <KeyboardAvoidingView 
        style={styles.chatContainer}
        behavior={Platform.OS === 'ios' ? 'padding' : 'height'}
      >
        {/* Messages */}
        <FlatList
          style={styles.messagesList}
          data={messages}
          keyExtractor={(item, idx) => `${item.sender}-${idx}`}
          showsVerticalScrollIndicator={false}
          contentContainerStyle={styles.messagesContent}
          renderItem={({ item, index }) => {
            const messageId = `${item.sender}-${index}`;
            return (
              <View style={[
                styles.messageContainer,
                item.sender === 'bot' ? styles.botMessage : styles.userMessage
              ]}>
                <View style={[
                  styles.messageBubble,
                  item.sender === 'bot' ? styles.botBubble : styles.userBubble
                ]}>
                  <View style={styles.messageHeader}>
                    <Text style={[
                      styles.senderLabel,
                      item.sender === 'bot' ? styles.botLabel : styles.userLabel
                    ]}>
                      {item.sender === "bot" ? " AgriBot" : " You"}
                    </Text>
                  </View>
                  <Text style={[
                    styles.messageText,
                    item.sender === 'bot' ? styles.botText : styles.userText
                  ]}>
                    {item.text}
                  </Text>

                  {item.sender === "bot" && (
                    <View style={styles.actionButtons}>
                      <TouchableOpacity
                        style={[
                          styles.actionButton,
                          styles.listenButton,
                          (ttsLoading === messageId || isPlaying) && styles.disabledButton
                        ]}
                        onPress={() => handleTTS(item.text, messageId)}
                        disabled={ttsLoading === messageId || isPlaying}
                      >
                        <Text style={styles.buttonText}>
                          {ttsLoading === messageId ? " Converting..." : " Listen"}
                        </Text>
                      </TouchableOpacity>

                      {isPlaying && (
                        <TouchableOpacity
                          style={[styles.actionButton, styles.stopButton]}
                          onPress={stopAudio}
                        >
                          <Text style={styles.buttonText}>
                            ⏹ Stop
                          </Text>
                        </TouchableOpacity>
                      )}
                    </View>
                  )}
                </View>
              </View>
            );
          }}
        />

        {/* Loading Indicator */}
        {loading && (
          <View style={styles.loadingContainer}>
            <ActivityIndicator size="small" color="#4caf50" />
            <Text style={styles.loadingText}>AgriBot is thinking...</Text>
          </View>
        )}

        {/* Input Area */}
        <View style={styles.inputContainer}>
          <View style={styles.inputWrapper}>
            <TextInput
              style={styles.textInput}
              value={question}
              onChangeText={setQuestion}
              placeholder="Ask me about farming, crops, diseases..."
              placeholderTextColor="#999"
              multiline
              maxLength={500}
              editable={!loading}
            />
            <TouchableOpacity
              style={[
                styles.sendButton,
                (!question.trim() || loading) && styles.sendButtonDisabled
              ]}
              onPress={sendQuestion}
              disabled={loading || !question.trim()}
            >
              <Text style={styles.sendButtonText}>
                {loading ? "🔄" : "➤"}
              </Text>
            </TouchableOpacity>
          </View>
        </View>
      </KeyboardAvoidingView>
    </SafeAreaView>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#f8f9fa',
  },
  header: {
    backgroundColor: '#2e7d32',
    paddingVertical: 20,
    paddingHorizontal: 20,
    borderBottomLeftRadius: 25,
    borderBottomRightRadius: 25,
    shadowColor: '#000',
    shadowOffset: {
      width: 0,
      height: 4,
    },
    shadowOpacity: 0.15,
    shadowRadius: 8,
    elevation: 8,
  },
  headerContent: {
    alignItems: 'center',
    paddingTop: Platform.OS === 'ios' ? 10 : 20,
  },
  headerTitle: {
    fontSize: 28,
    fontWeight: '700',
    color: '#ffffff',
    textAlign: 'center',
    letterSpacing: 0.5,
  },
  headerSubtitle: {
    fontSize: 16,
    color: '#c8e6c9',
    textAlign: 'center',
    marginTop: 4,
    fontWeight: '400',
  },
  chatContainer: {
    flex: 1,
    backgroundColor: '#f8f9fa',
  },
  messagesList: {
    flex: 1,
    paddingHorizontal: 16,
  },
  messagesContent: {
    paddingVertical: 20,
    flexGrow: 1,
  },
  messageContainer: {
    marginVertical: 8,
  },
  botMessage: {
    alignItems: 'flex-start',
  },
  userMessage: {
    alignItems: 'flex-end',
  },
  messageBubble: {
    maxWidth: '85%',
    borderRadius: 20,
    paddingVertical: 12,
    paddingHorizontal: 16,
    shadowColor: '#000',
    shadowOffset: {
      width: 0,
      height: 2,
    },
    shadowOpacity: 0.1,
    shadowRadius: 4,
    elevation: 3,
  },
  botBubble: {
    backgroundColor: '#e8f5e9',
    borderBottomLeftRadius: 4,
    borderWidth: 1,
    borderColor: '#4caf50',
  },
  userBubble: {
    backgroundColor: '#4caf50',
    borderBottomRightRadius: 4,
  },
  messageHeader: {
    marginBottom: 4,
  },
  senderLabel: {
    fontSize: 12,
    fontWeight: '600',
    textTransform: 'uppercase',
    letterSpacing: 0.5,
  },
  botLabel: {
    color: '#2e7d32',
  },
  userLabel: {
    color: '#ffffff',
    opacity: 0.9,
  },
  messageText: {
    fontSize: 16,
    lineHeight: 22,
    fontWeight: '400',
  },
  botText: {
    color: '#1b5e20',
  },
  userText: {
    color: '#ffffff',
  },
  actionButtons: {
    flexDirection: 'row',
    justifyContent: 'flex-start',
    marginTop: 12,
    gap: 8,
  },
  actionButton: {
    paddingVertical: 8,
    paddingHorizontal: 16,
    borderRadius: 20,
    alignItems: 'center',
    justifyContent: 'center',
    minWidth: 80,
    shadowColor: '#000',
    shadowOffset: {
      width: 0,
      height: 1,
    },
    shadowOpacity: 0.1,
    shadowRadius: 2,
    elevation: 2,
  },
  listenButton: {
    backgroundColor: '#2e7d32',
  },
  stopButton: {
    backgroundColor: '#f44336',
  },
  disabledButton: {
    backgroundColor: '#bdbdbd',
    opacity: 0.7,
  },
  buttonText: {
    color: '#ffffff',
    fontSize: 13,
    fontWeight: '600',
  },
  loadingContainer: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    paddingVertical: 16,
    paddingHorizontal: 20,
  },
  loadingText: {
    marginLeft: 12,
    fontSize: 16,
    color: '#666666',
    fontStyle: 'italic',
  },
  inputContainer: {
    backgroundColor: '#ffffff',
    paddingHorizontal: 16,
    paddingVertical: 12,
    borderTopWidth: 1,
    borderTopColor: '#e0e0e0',
    shadowColor: '#000',
    shadowOffset: {
      width: 0,
      height: -2,
    },
    shadowOpacity: 0.1,
    shadowRadius: 4,
    elevation: 8,
  },
  inputWrapper: {
    flexDirection: 'row',
    alignItems: 'flex-end',
    backgroundColor: '#f5f5f5',
    borderRadius: 25,
    paddingHorizontal: 4,
    paddingVertical: 4,
    borderWidth: 1,
    borderColor: '#e0e0e0',
  },
  textInput: {
    flex: 1,
    fontSize: 16,
    color: '#333333',
    paddingHorizontal: 16,
    paddingVertical: 12,
    maxHeight: 120,
    lineHeight: 22,
  },
  sendButton: {
    width: 44,
    height: 44,
    borderRadius: 22,
    backgroundColor: '#4caf50',
    alignItems: 'center',
    justifyContent: 'center',
    marginLeft: 8,
    shadowColor: '#000',
    shadowOffset: {
      width: 0,
      height: 2,
    },
    shadowOpacity: 0.15,
    shadowRadius: 4,
    elevation: 4,
  },
  sendButtonDisabled: {
    backgroundColor: '#bdbdbd',
    opacity: 0.6,
  },
  sendButtonText: {
    fontSize: 18,
    color: '#ffffff',
    fontWeight: '600',
  },
});
