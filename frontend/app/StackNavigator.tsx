import React from "react";
import { createNativeStackNavigator } from "@react-navigation/native-stack";
import { NavigationContainer } from "@react-navigation/native";

import Index from "./index";
import AnalysisPage from "./components/CropRecomendation";
import ChatScreen from "./ChatBot";
import Home from "./home";
import WelcomeScreen from "./WelcomeScreen";
import LoginScreen from "./LoginScreen";
import SignupScreen from "./SignupScreen";



export type RootStackParamList = {
  WelcomeScreen: undefined;
  LoginScreen: undefined;
  SignupScreen: undefined;
  Home: undefined;
  Analysis: undefined;
  ChatScreen: undefined;
};

const Stack = createNativeStackNavigator<RootStackParamList>();

const StackNavigator: React.FC = () => {
  return (
    <NavigationContainer>
      <Stack.Navigator initialRouteName="WelcomeScreen" screenOptions={{ headerShown: false }}>
        <Stack.Screen name="WelcomeScreen" component={WelcomeScreen} />
        <Stack.Screen name="LoginScreen" component={LoginScreen} />
        <Stack.Screen name="SignupScreen" component={SignupScreen} />
        <Stack.Screen name="Home" component={Home} />
        <Stack.Screen name="Analysis" component={AnalysisPage} />
        <Stack.Screen name="ChatScreen" component={ChatScreen} />
      </Stack.Navigator>
    </NavigationContainer>
  );
};

export default StackNavigator;
