import React, { useState } from 'react';
import { View, Text, TouchableOpacity, Image, Alert, StyleSheet } from 'react-native';
import { Camera, Upload, X } from "lucide-react-native";
import * as ImagePicker from 'expo-image-picker';

interface ImageUploadProps {
  onImageSelect: (file: any) => void;
  isLoading?: boolean;
}

const ImageUpload: React.FC<ImageUploadProps> = ({ onImageSelect, isLoading }) => {
  const [preview, setPreview] = useState<string | null>(null);

  const requestPermissions = async () => {
    const { status } = await ImagePicker.requestMediaLibraryPermissionsAsync();
    if (status !== 'granted') {
      Alert.alert('Permission Required', 'Sorry, we need camera roll permissions to make this work!');
      return false;
    }
    return true;
  };

  const requestCameraPermissions = async () => {
    const { status } = await ImagePicker.requestCameraPermissionsAsync();
    if (status !== 'granted') {
      Alert.alert('Permission Required', 'Sorry, we need camera permissions to take photos!');
      return false;
    }
    return true;
  };

  const pickImageFromGallery = async () => {
    const hasPermission = await requestPermissions();
    if (!hasPermission) return;

    const result = await ImagePicker.launchImageLibraryAsync({
      mediaTypes: ImagePicker.MediaTypeOptions.Images,
      allowsEditing: true,
      aspect: [4, 3],
      quality: 1,
    });

    if (!result.canceled && result.assets[0]) {
      const imageUri = result.assets[0].uri;
      setPreview(imageUri);
      onImageSelect(result.assets[0]);
    }
  };

  const takePhoto = async () => {
    const hasPermission = await requestCameraPermissions();
    if (!hasPermission) return;

    const result = await ImagePicker.launchCameraAsync({
      allowsEditing: true,
      aspect: [4, 3],
      quality: 1,
    });

    if (!result.canceled && result.assets[0]) {
      const imageUri = result.assets[0].uri;
      setPreview(imageUri);
      onImageSelect(result.assets[0]);
    }
  };

  const clearImage = () => {
    setPreview(null);
  };

  return (
    <View style={styles.container}>
      <View style={styles.header}>
        <Camera size={48} color="#4CAF50" style={styles.icon} />
        <Text style={styles.title}>Upload Crop Image</Text>
        <Text style={styles.subtitle}>
          Take a photo or upload an image of your crop for disease analysis
        </Text>
      </View>

      {preview ? (
        <View style={styles.previewContainer}>
          <Image source={{ uri: preview }} style={styles.previewImage} />
          <TouchableOpacity
            style={styles.clearButton}
            onPress={clearImage}
          >
            <X size={20} color="white" />
          </TouchableOpacity>
        </View>
      ) : (
        <View style={styles.uploadArea}>
          <Upload size={32} color="#999" style={styles.uploadIcon} />
          <Text style={styles.uploadText}>
            Tap the buttons below to take a photo or select from gallery
          </Text>
        </View>
      )}

      <View style={styles.buttonContainer}>
        <TouchableOpacity
          style={[styles.button, styles.cameraButton]}
          onPress={takePhoto}
          disabled={isLoading}
        >
          <Camera size={20} color="white" />
          <Text style={styles.buttonText}>Take Photo</Text>
        </TouchableOpacity>
        <TouchableOpacity
          style={[styles.button, styles.galleryButton]}
          onPress={pickImageFromGallery}
          disabled={isLoading}
        >
          <Upload size={20} color="white" />
          <Text style={styles.buttonText}>From Gallery</Text>
        </TouchableOpacity>
      </View>
    </View>
  );
};

const styles = StyleSheet.create({
  container: {
    backgroundColor: '#fff',
    borderRadius: 12,
    padding: 24,
    margin: 16,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.1,
    shadowRadius: 4,
    elevation: 3,
  },
  header: {
    alignItems: 'center',
    marginBottom: 24,
  },
  icon: {
    marginBottom: 8,
  },
  title: {
    fontSize: 18,
    fontWeight: 'bold',
    marginBottom: 8,
    textAlign: 'center',
  },
  subtitle: {
    fontSize: 14,
    color: '#666',
    textAlign: 'center',
    lineHeight: 20,
  },
  previewContainer: {
    position: 'relative',
    marginBottom: 24,
  },
  previewImage: {
    width: '100%',
    height: 192,
    borderRadius: 8,
  },
  clearButton: {
    position: 'absolute',
    top: 8,
    right: 8,
    backgroundColor: 'rgba(0,0,0,0.6)',
    borderRadius: 16,
    padding: 8,
  },
  uploadArea: {
    borderWidth: 2,
    borderStyle: 'dashed',
    borderColor: '#ddd',
    borderRadius: 8,
    padding: 32,
    alignItems: 'center',
    marginBottom: 24,
  },
  uploadIcon: {
    marginBottom: 8,
  },
  uploadText: {
    fontSize: 14,
    color: '#666',
    textAlign: 'center',
  },
  buttonContainer: {
    flexDirection: 'row',
    gap: 12,
  },
  button: {
    flex: 1,
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    padding: 16,
    borderRadius: 8,
  },
  cameraButton: {
    backgroundColor: '#4CAF50',
  },
  galleryButton: {
    backgroundColor: '#2196F3',
  },
  buttonText: {
    color: 'white',
    marginLeft: 8,
    fontSize: 16,
    fontWeight: '600',
  },
});

export default ImageUpload;