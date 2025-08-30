import React from "react";
import {
  View,
  Text,
  Image,
  ScrollView,
  StyleSheet,
} from "react-native";
import {
  Leaf,
  Brain,
  Mic,
  Camera,
  Globe,
  Zap,
} from "lucide-react-native";
import { router } from "expo-router";

import SimpleCard from "./components/SimpleCard";
import LanguageSelector from "./components/LanguageSelector";
import { useI18n } from "../lib/i18n";

// Import the hero image
const heroImage = require("../assets/images/farmImage.jpg");


const FeatureItem: React.FC<{ icon: React.ReactElement; text: string }> = ({ icon, text }) => (
  <View style={styles.featureItem}>
    {icon}
    <Text style={styles.featureText}>{text}</Text>
  </View>
);

const FeatureCard: React.FC<{ icon: React.ReactElement; title: string; description: string }> = ({
  icon,
  title,
  description,
}) => (
  <View style={styles.card}>
    <View style={styles.cardIcon}>{icon}</View>
    <Text style={styles.cardTitle}>{title}</Text>
    <Text style={styles.cardDescription}>{description}</Text>
  </View>
);

const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: "#fff" },
  heroSection: { position: "relative", height: 300 },
  heroImage: { width: "100%", height: "100%", position: "absolute" },
  heroOverlay: {
    ...StyleSheet.absoluteFillObject,
    backgroundColor: "rgba(0,0,0,0.4)",
  },
  heroContent: {
    flex: 1,
    justifyContent: "center",
    alignItems: "center",
    paddingHorizontal: 16,
  },
  heroTitleRow: { flexDirection: "row", alignItems: "center", marginBottom: 8 },
  heroTitle: { fontSize: 28, fontWeight: "bold", color: "#fff", marginLeft: 8 },
  heroSubtitle: { fontSize: 16, color: "#eee", textAlign: "center", marginVertical: 12 },
  featuresRow: { flexDirection: "row", flexWrap: "wrap", justifyContent: "center" },
  featureItem: { flexDirection: "row", alignItems: "center", margin: 6 },
  featureText: { color: "#fff", marginLeft: 6 },
  section: { padding: 16 },
  row: { flexDirection: "row", justifyContent: "space-around", margin: 16 },
  featuresSection: { backgroundColor: "#f4f4f4", padding: 20 },
  sectionTitle: { fontSize: 24, fontWeight: "bold", textAlign: "center", marginBottom: 8 },
  sectionSubtitle: { fontSize: 14, color: "#555", textAlign: "center", marginBottom: 20 },
  cardsRow: { flexDirection: "row", justifyContent: "space-between" },
  card: {
    flex: 1,
    backgroundColor: "#fff",
    borderRadius: 12,
    padding: 16,
    alignItems: "center",
    marginHorizontal: 6,
    elevation: 3, // Android shadow
    shadowColor: "#000", // iOS shadow
    shadowOpacity: 0.1,
    shadowRadius: 4,
    shadowOffset: { width: 0, height: 2 },
  },
  cardIcon: {
    backgroundColor: "#4CAF50",
    borderRadius: 50,
    padding: 12,
    marginBottom: 10,
  },
  cardTitle: { fontSize: 18, fontWeight: "600", marginBottom: 6 },
  cardDescription: { fontSize: 14, color: "#555", textAlign: "center" },
});
const Home: React.FC = () => {
  const { translations, currentLanguage, changeLanguage } = useI18n();

  return (
    <ScrollView style={styles.container}>
      {/* Hero Section */}
      <View style={styles.heroSection}>
        <Image source={heroImage} style={styles.heroImage} resizeMode="cover" />
        <View style={styles.heroOverlay} />
        <View style={styles.heroContent}>
          <View style={styles.heroTitleRow}>
            <Leaf size={32} color="#fff" />
            {/* <Image source={heroImage} style={styles.heroImage} resizeMode="cover" /> */}

            <Text style={styles.heroTitle}>{translations.heroTitle}</Text>
          </View>
          <Text style={styles.heroSubtitle}>
            {translations.heroSubtitle}
          </Text>
          <View style={styles.featuresRow}>
            <FeatureItem icon={<Camera size={18} color="#fff" />} text={translations.imageDetection} />
            <FeatureItem icon={<Mic size={18} color="#fff" />} text={translations.voiceOutput} />
            <FeatureItem icon={<Globe size={18} color="#fff" />} text={translations.multiLanguage} />
            <FeatureItem icon={<Zap size={18} color="#fff" />} text={translations.instantResults} />
          </View>
        </View>
      </View>

      {/* Language Selector */}
      <LanguageSelector
        selectedLanguage={currentLanguage}
        onLanguageChange={changeLanguage}
      />

      {/*  cards components */}

       <View style={{ flex: 1, justifyContent: "center" }}>
        <SimpleCard
          text={translations.cropRecommendations}
          buttonText={translations.getTips}
          imageSource={require("../assets/images/cropImage1.jpg")}
          onPress={() => router.push("/ChatBot")}
          
          // onPress={() => Alert.alert("Crop Diagnosis", "Feature coming soon!")}
        />
      </View>

      <View style={{ flex: 1, justifyContent: "center" }}>
        <SimpleCard
          text={translations.cropDiagnosis}
          buttonText={translations.go}
          imageSource={require("../assets/images/cropImage2.jpg")}
          onPress={() => router.push("/crop-diagnosis")}
        />
      </View>

      {/* Results */}
      {/* <DiagnosisResults
        results={diagnosisResults}
        isLoading={isAnalyzing}
        onPlayAudio={() => console.log("Playing audio response")}
      /> */}

      {/* Features Section */}
      <View style={styles.featuresSection}>
        {/* <Text style={styles.sectionTitle}>{translations.howItWorks}</Text> */}
        <Text style={styles.sectionTitle}>How KrishiSaarthi works</Text>
        <Text style={styles.sectionSubtitle}>
          {translations.howItWorksSubtitle}
        </Text>

        <View style={{ gap: 16 }}>
          <FeatureCard
            icon={<Camera size={32} color="#fff" />}
            title={translations.uploadAnalyze}
            description={translations.uploadAnalyzeDesc}
          />
          <FeatureCard
            icon={<Brain size={32} color="#fff" />}
            title={translations.aiDiagnosis}
            description={translations.aiDiagnosisDesc}
          />
          <FeatureCard
            icon={<Mic size={32} color="#fff" />}
            title={translations.voiceText}
            description={translations.voiceTextDesc}
          />
        </View>


      </View>
    </ScrollView>

  );
};

export default Home;
