import React, { useState, useEffect } from "react";
import {
  View,
  TouchableOpacity,
  Text,
  StyleSheet,
  Image,
  Dimensions,
} from "react-native";
import { Audio } from "expo-av";
import { Recording, Sound } from "expo-av/build/Audio";
import { Entypo, MaterialIcons } from "@expo/vector-icons";
import * as FileSystem from "expo-file-system";
import axios from "@/utils/axios";
import Animated, {
  useSharedValue,
  useAnimatedStyle,
  withRepeat,
  withTiming,
  Easing,
} from "react-native-reanimated";
import { LinearGradient } from "expo-linear-gradient";

const AnimatedMaterialIcons = Animated.createAnimatedComponent(MaterialIcons);
const screenWidth = Dimensions.get("window").width;

const RecordingButton = () => {
  const [recording, setRecording] = useState<null | Recording>(null);
  const [isRecording, setIsRecording] = useState(false);
  const [recordingStatus, setRecordingStatus] = useState("Ready to record");
  const [recordedURI, setRecordedURI] = useState<null | string>(null);
  const [sound, setSound] = useState<null | Sound>(null);
  const [isPlaying, setIsPlaying] = useState(false);
  const [uploadResult, setUploadResult] = useState<null | string>(null);
  const [uploadError, setUploadError] = useState<null | string>(null);
  const [result, setResult] = useState<any | null>(null);
  const [confidence, setConfidence] = useState<number>(0);

  const scale = useSharedValue(1);
  const opacity = useSharedValue(1);

  const progressBarColor = result?.severity === "" ? "#F44336" : "#4CAF50"; // green or red
  const progressBarWidth = Math.round(
    ((result?.updrs_score - 5) / (45 - 5)) * (screenWidth - 40)
  );

  const animatedStyles = useAnimatedStyle(() => {
    return {
      transform: [{ scale: scale.value }],
      opacity: opacity.value,
    };
  });

  async function startRecording() {
    try {
      scale.value = withRepeat(
        withTiming(1.2, {
          duration: 1000,
          easing: Easing.inOut(Easing.ease),
        }),
        -1,
        true
      );
      opacity.value = withRepeat(
        withTiming(0.7, {
          duration: 1000,
          easing: Easing.inOut(Easing.ease),
        }),
        -1,
        true
      );
      setResult(null);
      setUploadError(null);
      setUploadResult(null);
      console.log("Requesting permissions..");
      await Audio.requestPermissionsAsync();
      await Audio.setAudioModeAsync({
        allowsRecordingIOS: true,
        playsInSilentModeIOS: true,
      });

      console.log("Starting recording..");
      const { recording } = await Audio.Recording.createAsync({
        android: {
          extension: ".wav",
          outputFormat: Audio.AndroidOutputFormat.PCM_16BIT,
          audioEncoder: Audio.AndroidAudioEncoder.DEFAULT,
        },
        ios: {
          extension: ".wav",
          outputFormat: Audio.IOSOutputFormat.LINEARPCM,
          audioQuality: Audio.IOSAudioQuality.MAX,
          sampleRate: 44100,
          numberOfChannels: 1,
          bitRate: 128000,
        },
        web: {},
      });

      // const { recording } = await Audio.Recording.createAsync(
      //   Audio.RecordingOptionsPresets.HIGH_QUALITY
      // );

      setRecording(recording);
      setIsRecording(true);
      setRecordingStatus("Recording in progress...");
      console.log("Recording started");
    } catch (err) {
      scale.value = 1;
      opacity.value = 1;

      console.error("Failed to start recording", err);
      setRecordingStatus("Failed to start recording");
    }
  }

  async function uploadAudio(recordedURI: string) {
    if (!recordedURI) {
      setRecordingStatus("No recording to upload");
      return;
    }

    setRecordingStatus("Uploading...");

    try {
      const formData = new FormData();

      const fileInfo = await FileSystem.getInfoAsync(recordedURI);
      const fileType = fileInfo.uri.split(".").pop();

      // In React Native, we need to use a special format for FormData files
      // @ts-ignore - React Native's FormData implementation accepts this format
      formData.append("audio", {
        uri: recordedURI,
        name: `recording.${fileType}`,
        type: `audio/${fileType}`,
      });

      formData.append("timestamp", new Date().toISOString());

      const response = await axios.post("/predict-updrs/", formData, {
        headers: {
          "Content-Type": "multipart/form-data",
        },
        timeout: 5000,
      });

      setResult(response.data);

      console.log("Upload result:", response.data);
      setRecordingStatus("Upload successful!");
    } catch (error: any) {
      console.error("Upload error:", error);
      setRecordingStatus("Upload failed: " + error.message);
    }
  }

  async function stopRecording() {
    console.log("Stopping recording..");
    setIsRecording(false);
    scale.value = withTiming(1, { duration: 200 });
    opacity.value = withTiming(1, { duration: 200 });

    try {
      if (recording) {
        await recording.stopAndUnloadAsync();
        await Audio.setAudioModeAsync({
          allowsRecordingIOS: false,
        });

        const uri = recording.getURI();
        setRecordedURI(uri);
        setRecording(null);
        setRecordingStatus("Recording was successfull");
        uploadAudio(uri!);
        // setRecordingStatus("Recording stopped and stored at: " + uri);
        // console.log("Recording stopped and stored at", uri);
      }
    } catch (err) {
      console.error("Failed to stop recording", err);
      setRecordingStatus("Failed to stop recording");
    }
  }

  async function playSound() {
    if (!recordedURI) return;

    try {
      // Stop any currently playing sound
      if (sound) {
        await sound.unloadAsync();
      }

      // Create and play the new sound
      const { sound: newSound } = await Audio.Sound.createAsync(
        { uri: recordedURI },
        { shouldPlay: true }
      );

      setSound(newSound);
      setIsPlaying(true);

      // Set up event listener for when playback finishes
      newSound.setOnPlaybackStatusUpdate((playbackStatus) => {
        if (playbackStatus.didJustFinish) {
          setIsPlaying(false);
        }
      });

      await newSound.playAsync();
    } catch (error) {
      console.error("Error playing sound", error);
    }
  }

  async function stopSound() {
    if (sound) {
      await sound.stopAsync();
      setIsPlaying(false);
    }
  }

  useEffect(() => {
    return sound
      ? () => {
          console.log("Unloading Sound");
          sound.unloadAsync();
        }
      : undefined;
  }, [sound]);

  return (
    <LinearGradient
      colors={["#4076FF", "#ffffff"]}
      style={styles.container}
      start={{ x: 0.5, y: 0 }}
      end={{ x: 0.5, y: 0.4 }}
    >
      <View
        style={{
          position: "absolute",
          top: 0,
          width: "100%",
          flexDirection: "row",
          alignItems: "center",
          padding: 10,
          gap: 10,
          backgroundColor: "transparent",
        }}
      >
        <Image
          source={require("@/assets/images/parka.png")}
          style={{ width: 28, height: 28 }}
        />
        <Text style={{ fontSize: 24, fontWeight: 600, color: "white" }}>
          Parka
        </Text>
      </View>
      <Text
        style={{
          position: "absolute",
          left: 10,
          color: "white",
          top: 60,
          fontSize: 36,
          fontWeight: 500,
        }}
      >
        Parkinson Disease Detector
      </Text>
      <Text
        style={{
          position: "absolute",
          left: 10,
          color: "white",
          top: 160,
          fontSize: 18,
          fontWeight: 500,
        }}
      >
        Press the button below to see if you have parkison
      </Text>
      <TouchableOpacity
        onPress={isRecording ? stopRecording : startRecording}
        style={{
          display: "flex",
          justifyContent: "center",
          alignItems: "center",
        }}
      >
        {isRecording ? (
          <>
            <Text
              style={{
                position: "absolute",
                top: -50,
                fontSize: 24,
                textAlign: "center",
              }}
            >
              Say "AAAAA"
            </Text>
            <AnimatedMaterialIcons
              name="keyboard-voice"
              size={40}
              color="white"
              style={[
                {
                  backgroundColor: "red",
                  padding: 16,
                  borderRadius: 50,
                },
                animatedStyles,
              ]}
            />
          </>
        ) : (
          <MaterialIcons
            name="keyboard-voice"
            size={40}
            color="black"
            style={{
              padding: 16,
              backgroundColor: "#00000010",
              borderRadius: 50,
            }}
          />
        )}
      </TouchableOpacity>

      {recordedURI && (
        <TouchableOpacity
          onPress={isPlaying ? stopSound : playSound}
          style={{ position: "absolute", right: 20, bottom: 20 }}
        >
          {isPlaying ? (
            <Entypo
              name="controller-stop"
              size={24}
              color="white"
              style={{
                padding: 16,
                backgroundColor: "blue",
                borderRadius: 50,
              }}
            />
          ) : (
            <Entypo
              name="controller-play"
              size={24}
              color="black"
              style={{
                padding: 16,
                backgroundColor: "#00000010",
                borderRadius: 50,
              }}
            />
          )}
        </TouchableOpacity>
      )}

      <Text style={styles.statusText}>{uploadResult}</Text>
      {result && (
        <View
          style={{
            paddingHorizontal: 20,
            width: "100%",
            position: "absolute",
            bottom: 100,
            paddingBottom: 20,
          }}
        >
          <Text style={styles.resultText}>
            Parkinson: {result?.severity} (
            {Math.round(((result?.updrs_score - 5) / (45 - 5)) * 100)}%)
          </Text>
          <View style={styles.progressBarBackground}>
            <Animated.View
              style={[
                styles.progressBarFill,
                {
                  width: progressBarWidth,
                  backgroundColor: progressBarColor,
                },
              ]}
            />
          </View>
          <Text style={{ paddingVertical: 10, fontSize: 16, padding: 10, backgroundColor: "#00000010", marginTop: 10 }}>
            {result?.explanation}
          </Text>
        </View>
      )}
    </LinearGradient>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
    justifyContent: "center",
    alignItems: "center",
    gap: 20,
    backgroundColor: "white",
  },
  button: {
    padding: 15,
    borderRadius: 50,
    width: 200,
    alignItems: "center",
    marginBottom: 20,
  },
  recordingButton: {
    backgroundColor: "#f44336",
  },
  playButton: {
    backgroundColor: "#2196F3",
  },
  playingButton: {
    backgroundColor: "#FF9800",
  },
  buttonText: {
    color: "white",
    fontSize: 18,
    fontWeight: "bold",
  },
  statusText: {
    fontSize: 16,
    color: "#333",
    textAlign: "center",
  },
  resultText: {
    marginTop: 20,
    fontSize: 18,
    fontWeight: "600",
    textAlign: "center",
  },
  progressBarBackground: {
    width: "100%",
    height: 16,
    backgroundColor: "#eee",
    borderRadius: 8,
    marginTop: 10,
    overflow: "hidden",
  },
  progressBarFill: {
    height: "100%",
    borderRadius: 8,
    padding: 10,
  },
});

export default RecordingButton;
