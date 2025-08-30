import React from 'react';
import {
  View,
  Text,
  TouchableOpacity,
  StyleSheet,
  SafeAreaView,
  Alert,
} from 'react-native';
import { router, Link } from 'expo-router';

const WelcomeScreen = () => {
  return (
    <SafeAreaView style={styles.container}>
      <View style={styles.content}>
        <View style={styles.headerSection}>
          <Text style={styles.appTitle}> KrishiSaarthi </Text>
          <Text style={styles.tagline}>Your Smart Farming Companion</Text>
          <Text style={styles.description}>
            Get intelligent farming advice, weather and crop recommendations with our AI-powered assistant.
          </Text>
        </View>

        <View style={styles.buttonSection}>
          <TouchableOpacity
            style={styles.loginButton}
            onPress={() => {
              console.log('Login button pressed');
              router.push('/LoginScreen');
            }}
          >
            <Text style={styles.loginButtonText}>Login</Text>
          </TouchableOpacity>

          <TouchableOpacity
            style={styles.signupButton}
            onPress={() => {
              Alert.alert('Button Pressed!', 'Create Account button was pressed');
              console.log('Create Account button pressed');
              try {
                router.push('/SignupScreen');
              } catch (error) {
                console.error('Navigation error:', error);
                Alert.alert('Navigation Error', error.toString());
              }
            }}
          >
            <Text style={styles.signupButtonText}>Create Account</Text>
          </TouchableOpacity>

          {/* Alternative Link approach for testing */}

        </View>
      </View>
    </SafeAreaView>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#f8f9fa',
  },
  content: {
    flex: 1,
    paddingLeft: 24,
    paddingRight: 24,
    paddingTop: 32,
    paddingBottom: 32,
    justifyContent: 'space-between',
  },
  headerSection: {
    alignItems: 'center',
    marginTop: 80,
  },
  appTitle: {
    fontSize: 36,
    fontWeight: 'bold',
    color: '#2e7d32',
    marginBottom: 16,
    textAlign: 'center',
  },
  tagline: {
    fontSize: 20,
    fontWeight: 'bold',
    color: '#1b5e20',
    marginBottom: 24,
    textAlign: 'center',
  },
  description: {
    fontSize: 16,
    color: '#666',
    textAlign: 'center',
    lineHeight: 24,
    paddingLeft: 16,
    paddingRight: 16,
  },
  buttonSection: {
    marginBottom: 40,
  },
  loginButton: {
    backgroundColor: '#2e7d32',
    paddingTop: 16,
    paddingBottom: 16,
    paddingLeft: 16,
    paddingRight: 16,
    borderRadius: 12,
    alignItems: 'center',
    marginBottom: 16,
  },
  loginButtonText: {
    color: '#fff',
    fontSize: 18,
    fontWeight: 'bold',
  },
  signupButton: {
    backgroundColor: '#fff',
    paddingTop: 16,
    paddingBottom: 16,
    paddingLeft: 16,
    paddingRight: 16,
    borderRadius: 12,
    alignItems: 'center',
    borderWidth: 2,
    borderColor: '#2e7d32',
    marginBottom: 24,
  },
  signupButtonText: {
    color: '#2e7d32',
    fontSize: 18,
    fontWeight: 'bold',
  },
  dividerContainer: {
    flexDirection: 'row',
    alignItems: 'center',
    marginTop: 20,
    marginBottom: 20,
  },
  divider: {
    flex: 1,
    height: 1,
    backgroundColor: '#ddd',
  },
  dividerText: {
    paddingLeft: 16,
    paddingRight: 16,
    color: '#666',
    fontSize: 14,
  },
  infoText: {
    fontSize: 14,
    color: '#666',
    textAlign: 'center',
    lineHeight: 20,
  },
});

export default WelcomeScreen;
