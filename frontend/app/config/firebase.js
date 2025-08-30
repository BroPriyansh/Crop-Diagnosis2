import { initializeApp } from 'firebase/app';
import { getAuth, initializeAuth, getReactNativePersistence } from 'firebase/auth';
import AsyncStorage from '@react-native-async-storage/async-storage';

// Your web app's Firebase configuration
const firebaseConfig = {
  apiKey: "AIzaSyC0LTLskz7ET8p7pL3EEFHPKziaqcvSYbs",
  authDomain: "krsh-daeff.firebaseapp.com",
  projectId: "krsh-daeff",
  storageBucket: "krsh-daeff.firebasestorage.app",
  messagingSenderId: "700557136106",
  appId: "1:700557136106:web:858494309e148aceaa4793"
};

// Initialize Firebase
const app = initializeApp(firebaseConfig);

// Initialize Firebase Auth with React Native persistence
let auth;
try {
  auth = initializeAuth(app, {
    persistence: getReactNativePersistence(AsyncStorage)
  });
} catch (error) {
  // If auth is already initialized, get the existing instance
  auth = getAuth(app);
}

export { auth };
export default app;
