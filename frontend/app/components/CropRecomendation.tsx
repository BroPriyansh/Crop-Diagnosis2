import React, { useState } from "react";
import { View, ScrollView, StyleSheet, Text, ActivityIndicator } from "react-native";
import ImageUpload from "./ImageUpload";
import VoiceRecorder from "./VoiceRecorder";
import DiagnosisResults from "./DiagnosisResults";

const AnalysisPage: React.FC = () => {
  const [diagnosisResults, setDiagnosisResults] = useState<any>(null);
  const [isAnalyzing, setIsAnalyzing] = useState<boolean>(false);

  // Mock image analysis
  const handleImageSelect = async (file: any) => {
    setIsAnalyzing(true);
    setTimeout(() => {
      setDiagnosisResults({
        disease: "Tomato Late Blight",
        confidence: 0.89,
        severity: "medium",
        description:
          "Late blight is a fungal disease affecting tomato plants. Take immediate action.",
        recommendations: [
          "Apply copper-based fungicide",
          "Remove affected leaves",
          "Improve air circulation",
        ],
      });
      setIsAnalyzing(false);
    }, 2000);
  };

  // Mock audio analysis
  const handleAudioRecorded = async (audioBlob: any) => {
    setIsAnalyzing(true);
    setTimeout(() => {
      setDiagnosisResults({
        disease: "General Crop Advice",
        confidence: 0.95,
        severity: "low",
        description: "Recommendations based on your voice query about crop health.",
        recommendations: [
          "Test soil pH regularly",
          "Apply balanced fertilizer",
          "Rotate crops yearly",
        ],
      });
      setIsAnalyzing(false);
    }, 2000);
  };

  return (
    <ScrollView style={styles.container}>
      {/* Input Section */}
      <View style={styles.inputSection}>
        <Text style={styles.sectionTitle}>Analyze Your Crop</Text>
        <View style={styles.cardsRow}>
          <ImageUpload onImageSelect={handleImageSelect} isLoading={isAnalyzing} />
          <VoiceRecorder onAudioRecorded={handleAudioRecorded} isLoading={isAnalyzing} />
        </View>
        {isAnalyzing && <ActivityIndicator size="large" color="#4CAF50" style={{ marginTop: 16 }} />}
      </View>

      {/* Results Section */}
      {diagnosisResults && (
        <View style={styles.resultsSection}>
          <Text style={styles.sectionTitle}>Diagnosis Results</Text>
          <DiagnosisResults
            results={diagnosisResults}
            isLoading={isAnalyzing}
            onPlayAudio={() => console.log("Playing audio response")}
          />
        </View>
      )}
    </ScrollView>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: "#f4f4f4",
    paddingVertical: 16,
  },
  inputSection: {
    backgroundColor: "#fff",
    marginHorizontal: 16,
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
    marginTop: 16,
  },
});

export default AnalysisPage;
