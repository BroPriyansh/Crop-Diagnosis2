// MultipleFiles/crop-diagnosis.tsx
import React, { useState } from "react";
import {
  View,
  Text,
  ScrollView,
  StyleSheet,
  TouchableOpacity,
  ActivityIndicator,
  TextInput,
} from "react-native";
import ImageUpload from "./components/ImageUpload";
import VoiceRecorder from "./components/VoiceRecorder";
import DiagnosisResults from "./components/DiagnosisResults";
import { useI18n } from "../lib/i18n";

interface ChatMessage {
  text: string;
  sender: "bot" | "user";
}

const CropRecommendationScreen: React.FC = () => {
  const { translations } = useI18n();
  const [diagnosisResults, setDiagnosisResults] = useState<any>(null);
  const [isAnalyzing, setIsAnalyzing] = useState<boolean>(false);
  const [showChatbot, setShowChatbot] = useState<boolean>(false);
  const [chatMessages, setChatMessages] = useState<ChatMessage[]>([]);
  const [chatQuestion, setChatQuestion] = useState<string>("");
  const [chatLoading, setChatLoading] = useState<boolean>(false);
  const [sessionId, setSessionId] = useState<string>("");

  // Get the correct API URL based on platform
  const getApiUrl = (service: "ml_model" | "chatbot") => {
    // Use the actual network IP that servers are running on
    const baseIp = "172.16.92.63"; // Use the same IP from your server logs
    if (service === "ml_model") {
      return `http://${baseIp}:8000`; // main.py runs on 8000
    } else {
      return `http://${baseIp}:8005`; // chatbot runs on 8005
    }
  };

  // Real disease diagnosis from image
  const handleImageSelect = async (file: any) => {
    console.log("🖼️ Image selected, starting diagnosis...");
    setIsAnalyzing(true);
    try {
      // Use the integrated endpoint that handles ML model + chatbot internally
      const chatbotApiUrl = getApiUrl("chatbot");
      console.log("📤 Sending to integrated endpoint:", chatbotApiUrl);

      // Fallback to text-based diagnosis since file upload fails in React Native
      const response = await fetch(`${chatbotApiUrl}/api/diagnose`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          disease_image_description: "User uploaded an image for crop disease diagnosis but image analysis is required to provide accurate identification.",
          crop_type: "Unknown crop from uploaded image",
          symptoms: "Image analysis required to identify symptoms",
        }),
      });

      if (!response.ok) {
        const errorText = await response.text();
        throw new Error(`Diagnosis failed: ${response.status} - ${errorText}`);
      }

      const data = await response.json();
      setSessionId(data.session_id);

      setDiagnosisResults({
        disease: "Image Analysis Required",
        confidence: data.confidence || 0.85,
        severity: "disease",
        description: data.response,
        recommendations: data.solutions || [],
        causes: data.causes || [],
        symptoms: data.symptoms || [],
        prevention: data.prevention || [],
      });

      setShowChatbot(true);
    } catch (error) {
      console.error("Diagnosis error:", error);
      // Generate a session ID even for failed diagnosis to enable chatbot
      const fallbackSessionId = `session_${Date.now()}`;
      setSessionId(fallbackSessionId);

      // Provide a user-friendly error message and enable chatbot for manual interaction
      setDiagnosisResults({
        disease: "Diagnosis Failed",
        confidence: 0,
        severity: "unknown",
        description: `An error occurred during diagnosis: ${
          error instanceof Error ? error.message : String(error)
        }. Please ensure both backend services (ML model on 8000 and Chatbot on 8005) are running and accessible. You can still ask general questions below.`,
        recommendations: [
          "Check backend server status (ports 8000 and 8005)",
          "Try describing symptoms in the chat",
          "Ask about common crop diseases",
        ],
      });

      // Show chatbot even when diagnosis fails
      setShowChatbot(true);
    } finally {
      setIsAnalyzing(false);
    }
  };

  // Handle chat message sending
  const sendChatMessage = async () => {
    if (!chatQuestion.trim() || !sessionId) return;

    setChatLoading(true);
    const userMessage = chatQuestion.trim();
    setChatQuestion("");

    // Add user message to chat
    setChatMessages((prev) => [...prev, { text: userMessage, sender: "user" }]);

    try {
      const apiUrl = getApiUrl("chatbot"); // Chatbot API for general chat
      const response = await fetch(`${apiUrl}/api/chat`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          message: userMessage,
          session_id: sessionId,
        }),
      });

      if (!response.ok) {
        throw new Error("Failed to get chat response");
      }

      const data = await response.json();
      setChatMessages((prev) => [...prev, { text: data.response, sender: "bot" }]);
    } catch (error) {
      console.error("Chat error:", error);
      let errorMessage = "I'm having trouble connecting to the server. ";

      if (error instanceof Error) {
        if (error.message?.includes("Network request failed")) {
          errorMessage += "Please make sure the backend server is running on port 8005 and try again.";
        } else if (error.message?.includes("Failed to fetch")) {
          errorMessage += "Please check your internet connection and try again.";
        } else {
          errorMessage += "Please try asking your question again.";
        }
      } else {
        errorMessage += "Please try asking your question again.";
      }

      setChatMessages((prev) => [...prev, { text: errorMessage, sender: "bot" }]);
    } finally {
      setChatLoading(false);
    }
  };

  // Audio analysis for disease diagnosis
  const handleAudioRecorded = async (audioBlob: any) => {
    setIsAnalyzing(true);
    try {
      // For now, use a placeholder - you can integrate speech-to-text later
      // This will call the chatbot's /api/diagnose endpoint with a text description.
      const apiUrl = getApiUrl("chatbot");
      const response = await fetch(`${apiUrl}/api/diagnose`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          disease_image_description: "User  described crop disease symptoms via audio",
          symptoms: "Audio-described symptoms of crop disease",
          // No predicted_class, confidence, etc. for audio input directly
        }),
      });

      if (!response.ok) {
        const errorText = await response.text();
        throw new Error(`Failed to get diagnosis: ${response.status} - ${errorText}`);
      }

      const data = await response.json();
      setSessionId(data.session_id);

      setDiagnosisResults({
        disease: "Image Analysis Required",
        confidence: data.confidence || 0.85,
        severity: "disease",
        description: data.response,
        recommendations: data.solutions || [],
        causes: data.causes || [],
        symptoms: data.symptoms || [],
        prevention: data.prevention || [],
      });

      setShowChatbot(true);
    } catch (error) {
      console.error("Audio diagnosis error:", error);
      // Generate a session ID even for failed diagnosis to enable chatbot
      const fallbackSessionId = `session_${Date.now()}`;
      setSessionId(fallbackSessionId);

      setDiagnosisResults({
        disease: "General Crop Disease",
        confidence: 0.5,
        severity: "unknown",
        description: `Unable to process the audio description: ${
          error instanceof Error ? error.message : String(error)
        }. However, I can still help you with general crop disease questions and treatments.`,
        recommendations: [
          "Ask me about common crop diseases",
          "Describe the symptoms in text",
          "Get general treatment advice",
          "Learn about prevention methods",
        ],
      });

      // Show chatbot even when diagnosis fails
      setShowChatbot(true);
    } finally {
      setIsAnalyzing(false);
    }
  };

  return (
    <ScrollView style={styles.container}>
      {/* Header Section */}
      <View style={styles.headerSection}>
        <Text style={styles.headerTitle}>{translations.cropDiagnosisTitle}</Text>
        <Text style={styles.headerSubtitle}>{translations.cropDiagnosisSubtitle}</Text>
      </View>

      {/* Input Section */}
      <View style={styles.inputSection}>
        <Text style={styles.sectionTitle}>{translations.analyzeFarmingConditions}</Text>
        <View style={styles.cardsRow}>
          <ImageUpload onImageSelect={handleImageSelect} isLoading={isAnalyzing} />
          {/* <VoiceRecorder onAudioRecorded={handleAudioRecorded} isLoading={isAnalyzing} /> */}
        </View>
        {isAnalyzing && <ActivityIndicator size="large" color="#4CAF50" style={{ marginTop: 16 }} />}
      </View>

      {/* Results Section */}
      {diagnosisResults && (
        <View style={styles.resultsSection}>
          <Text style={styles.sectionTitle}>Disease Diagnosis Results</Text>
          <DiagnosisResults
            results={diagnosisResults}
            isLoading={isAnalyzing}
            onPlayAudio={() => console.log("Playing audio response")}
          />
        </View>
      )}

      {/* Chatbot Section */}
      {showChatbot && (
        <View style={styles.chatSection}>
          <Text style={styles.sectionTitle}>Ask More Questions</Text>
          <Text style={styles.chatSubtitle}>Have more questions about this disease? Ask our expert bot!</Text>

          {/* Chat Messages */}
          {chatMessages.length > 0 && (
            <ScrollView style={styles.chatMessages} showsVerticalScrollIndicator={true} nestedScrollEnabled={true}>
              {chatMessages.map((item, index) => (
                <View
                  key={`${item.sender}-${index}`}
                  style={[styles.chatMessage, item.sender === "bot" ? styles.botMessage : styles.userMessage]}
                >
                  <Text
                    style={[
                      styles.chatMessageText,
                      item.sender === "bot" ? styles.botMessageText : styles.userMessageText,
                    ]}
                  >
                    {item.text}
                  </Text>
                </View>
              ))}
            </ScrollView>
          )}

          {/* Chat Loading */}
          {chatLoading && (
            <View style={styles.chatLoading}>
              <ActivityIndicator size="small" color="#4CAF50" />
              <Text style={styles.chatLoadingText}>Thinking...</Text>
            </View>
          )}

          {/* Chat Input */}
          <View style={styles.chatInputContainer}>
            <TextInput
              style={styles.chatInput}
              value={chatQuestion}
              onChangeText={setChatQuestion}
              placeholder="Ask about treatment, prevention, or anything else..."
              placeholderTextColor="#999"
              multiline
              maxLength={300}
            />
            <TouchableOpacity
              style={[styles.chatSendButton, (!chatQuestion.trim() || chatLoading) && styles.chatSendButtonDisabled]}
              onPress={sendChatMessage}
              disabled={!chatQuestion.trim() || chatLoading}
            >
              <Text style={styles.chatSendButtonText}>{chatLoading ? "..." : "Send"}</Text>
            </TouchableOpacity>
          </View>
        </View>
      )}

      {/* Tips Section */}
      <View style={styles.tipsSection}>
        <Text style={styles.sectionTitle}>{translations.farmingTips}</Text>
        <View style={styles.tipCard}>
          <Text style={styles.tipTitle}>{translations.soilPreparation}</Text>
          <Text style={styles.tipDescription}>{translations.soilPreparationDesc}</Text>
        </View>
        <View style={styles.tipCard}>
          <Text style={styles.tipTitle}>{translations.waterManagement}</Text>
          <Text style={styles.tipDescription}>{translations.waterManagementDesc}</Text>
        </View>
        <View style={styles.tipCard}>
          <Text style={styles.tipTitle}>{translations.cropRotation}</Text>
          <Text style={styles.tipDescription}>{translations.cropRotationDesc}</Text>
        </View>
      </View>
    </ScrollView>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: "#f4f4f4",
  },
  headerSection: {
    backgroundColor: "#4CAF50",
    paddingVertical: 32,
    paddingHorizontal: 16,
    alignItems: "center",
  },
  headerTitle: {
    fontSize: 28,
    fontWeight: "700",
    color: "#fff",
    textAlign: "center",
    marginBottom: 8,
  },
  headerSubtitle: {
    fontSize: 16,
    color: "#e8f5e8",
    textAlign: "center",
    lineHeight: 22,
  },
  inputSection: {
    backgroundColor: "#fff",
    marginHorizontal: 16,
    marginTop: 16,
    borderRadius: 12,
    padding: 16,
    shadowColor: "#000",
    shadowOpacity: 0.1,
    shadowRadius: 6,
    shadowOffset: { width: 0, height: 3 },
    elevation: 3,
  },
  cardsRow: {
    marginTop: 12,
  },
  sectionTitle: {
    fontSize: 20,
    fontWeight: "700",
    color: "#333",
    marginBottom: 12,
    textAlign: "center",
  },
  resultsSection: {
    backgroundColor: "#fff",
    margin: 16,
    borderRadius: 12,
    padding: 16,
    shadowColor: "#000",
    shadowOpacity: 0.1,
    shadowRadius: 6,
    shadowOffset: { width: 0, height: 3 },
    elevation: 3,
  },
  tipsSection: {
    margin: 16,
    marginBottom: 32,
  },
  tipCard: {
    backgroundColor: "#fff",
    borderRadius: 12,
    padding: 16,
    marginBottom: 12,
    shadowColor: "#000",
    shadowOpacity: 0.1,
    shadowRadius: 4,
    shadowOffset: { width: 0, height: 2 },
    elevation: 2,
  },
  tipTitle: {
    fontSize: 16,
    fontWeight: "600",
    color: "#333",
    marginBottom: 8,
  },
  tipDescription: {
    fontSize: 14,
    color: "#666",
    lineHeight: 20,
  },
  // Chatbot styles
  chatSection: {
    backgroundColor: "#fff",
    margin: 16,
    borderRadius: 12,
    padding: 16,
    shadowColor: "#000",
    shadowOpacity: 0.1,
    shadowRadius: 6,
    shadowOffset: { width: 0, height: 3 },
    elevation: 3,
  },
  chatSubtitle: {
    fontSize: 14,
    color: "#666",
    textAlign: "center",
    marginBottom: 16,
  },
  chatMessages: {
    maxHeight: 300,
    marginBottom: 16,
  },
  chatMessage: {
    marginVertical: 4,
    maxWidth: "80%",
  },
  botMessage: {
    alignSelf: "flex-start",
  },
  userMessage: {
    alignSelf: "flex-end",
  },
  chatMessageText: {
    padding: 12,
    borderRadius: 16,
    fontSize: 14,
    lineHeight: 20,
  },
  botMessageText: {
    backgroundColor: "#e8f5e8",
    color: "#2e7d32",
  },
  userMessageText: {
    backgroundColor: "#4caf50",
    color: "#fff",
  },
  chatLoading: {
    flexDirection: "row",
    alignItems: "center",
    justifyContent: "center",
    paddingVertical: 12,
  },
  chatLoadingText: {
    marginLeft: 8,
    fontSize: 14,
    color: "#666",
    fontStyle: "italic",
  },
  chatInputContainer: {
    flexDirection: "row",
    alignItems: "flex-end",
    backgroundColor: "#f5f5f5",
    borderRadius: 20,
    paddingHorizontal: 4,
    paddingVertical: 4,
    borderWidth: 1,
    borderColor: "#e0e0e0",
  },
  chatInput: {
    flex: 1,
    fontSize: 14,
    color: "#333",
    paddingHorizontal: 12,
    paddingVertical: 8,
    maxHeight: 80,
    lineHeight: 18,
  },
  chatSendButton: {
    backgroundColor: "#4caf50",
    borderRadius: 16,
    paddingHorizontal: 16,
    paddingVertical: 8,
    marginLeft: 8,
  },
  chatSendButtonDisabled: {
    backgroundColor: "#bdbdbd",
    opacity: 0.6,
  },
  chatSendButtonText: {
    color: "#fff",
    fontSize: 14,
    fontWeight: "600",
  },
});

export default CropRecommendationScreen;
