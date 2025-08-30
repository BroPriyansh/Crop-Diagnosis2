import React from "react";
import { View, Text, TouchableOpacity, StyleSheet, ImageBackground, GestureResponderEvent } from "react-native";

type SimpleCardProps = {
  text: string;
  buttonText: string;
  onPress: (event: GestureResponderEvent) => void;
  imageSource: any; // required background image
};

const SimpleCard: React.FC<SimpleCardProps> = ({ text, buttonText, onPress, imageSource }) => {
  return (
    <TouchableOpacity style={styles.cardContainer} onPress={onPress} activeOpacity={0.8}>
      <ImageBackground source={imageSource} style={styles.card} imageStyle={{ borderRadius: 12 }}>
        <View style={styles.overlay} />
        <View style={styles.content}>
          <Text style={styles.text}>{text}</Text>
          <View style={styles.button}>
            <Text style={styles.buttonText}>{buttonText}</Text>
          </View>
        </View>
      </ImageBackground>
    </TouchableOpacity>
  );
};

const styles = StyleSheet.create({
  cardContainer: {
    marginVertical: 8,
    width: "90%",
    alignSelf: "center",
  },
  card: {
    height: 250,
    justifyContent: "center",
    paddingHorizontal: 16,
    borderRadius: 12,
    overflow: "hidden",
  },
  overlay: {
    ...StyleSheet.absoluteFillObject,
    backgroundColor: "rgba(0,0,0,0.35)", // dark overlay for text readability
    borderRadius: 12,
  },
  content: {
    flexDirection: "row",
    justifyContent: "space-between",
    alignItems: "center",
  },
  text: {
    fontSize: 18,
    fontWeight: "bold",
    color: "#fff",
  },
  button: {
    backgroundColor: "#4CAF50",
    paddingVertical: 6,
    paddingHorizontal: 16,
    borderRadius: 8,
  },
  buttonText: {
    color: "#fff",
    fontWeight: "600",
    fontSize: 14,
  },
});

export default SimpleCard;
