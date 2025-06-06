import React, { useState, useEffect } from "react";
import {
  View,
  TouchableOpacity,
  Text,
  StyleSheet,
  Image,
  Dimensions,
  StatusBar,
  Pressable,
  ActivityIndicator,
} from "react-native";
import { Audio } from "expo-av";
import { Recording, Sound } from "expo-av/build/Audio";
import {
  Entypo,
  MaterialIcons,
  Ionicons,
  FontAwesome5,
} from "@expo/vector-icons";
import * as FileSystem from "expo-file-system";
import axios from "@/utils/axios";
import Animated, {
  useSharedValue,
  useAnimatedStyle,
  withRepeat,
  withTiming,
  Easing,
  interpolate,
  withSpring,
} from "react-native-reanimated";
import { LinearGradient } from "expo-linear-gradient";
import DrawParkinsonSpiral from "@/components/DrawParkinsonSpiral";
import { Drawer } from "react-native-drawer-layout";
import { useNavigation } from "@react-navigation/native";
import DoctorMap from "@/components/DoctorMap";
import { useRouter } from "expo-router";

const AnimatedMaterialIcons = Animated.createAnimatedComponent(MaterialIcons);
const { width: screenWidth, height: screenHeight } = Dimensions.get("window");

const STEPS = [
  {
    id: "cough",
    title: "Cough Detection",
    instruction: "Take a deep breath and cough naturally",
  },
  {
    id: "parkinson",
    title: "Parkinson's Detection",
    instruction: "Say 'AAAAA' for 5 seconds",
  },
];

const RecordingButton = () => {
  const [currentStep, setCurrentStep] = useState(0);
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
  const [open, setOpen] = useState(false);
  const [loading, setLoading] = useState(false);
  const navigation = useNavigation();
  const router = useRouter();

  const scale = useSharedValue(1);
  const opacity = useSharedValue(1);
  const waveAnimation1 = useSharedValue(0);
  const waveAnimation2 = useSharedValue(0);
  const waveAnimation3 = useSharedValue(0);
  const floatingAnimation = useSharedValue(0);

  const progressBarColor = result?.severity === "" ? "#F44336" : "#4CAF50";
  const progressBarWidth = Math.round(
    ((result?.updrs_score - 5) / (45 - 5)) * (screenWidth - 40)
  );

  const animatedStyles = useAnimatedStyle(() => {
    return {
      transform: [{ scale: scale.value }],
      opacity: opacity.value,
    };
  });

  useEffect(() => {
    waveAnimation1.value = withRepeat(
      withTiming(1, {
        duration: 4000,
        easing: Easing.inOut(Easing.ease),
      }),
      -1,
      true
    );
    waveAnimation2.value = withRepeat(
      withTiming(1, {
        duration: 3500,
        easing: Easing.inOut(Easing.ease),
      }),
      -1,
      true
    );
    waveAnimation3.value = withRepeat(
      withTiming(1, {
        duration: 3000,
        easing: Easing.inOut(Easing.ease),
      }),
      -1,
      true
    );
    floatingAnimation.value = withRepeat(
      withTiming(1, {
        duration: 2000,
        easing: Easing.inOut(Easing.ease),
      }),
      -1,
      true
    );
  }, []);

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
    setLoading(true);

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

      const response = await axios.post("/cough/", formData, {
        headers: {
          "Content-Type": "multipart/form-data",
        },
      });

      setResult(response.data);

      console.log("Upload result:", response.data);
      setRecordingStatus("Upload successful!");
    } catch (error: any) {
      console.error("Upload error:", error);
      setRecordingStatus("Upload failed: " + error.message);
    } finally {
      setLoading(false);
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

  const handleNext = () => {
    if (currentStep < STEPS.length - 1) {
      setCurrentStep(currentStep + 1);
      setResult(null);
      setRecordedURI(null);
    }
  };

  const handlePrevious = () => {
    if (currentStep > 0) {
      setCurrentStep(currentStep - 1);
      setResult(null);
      setRecordedURI(null);
    }
  };

  const handleBack = () => {
    if (currentStep > 0) {
      setCurrentStep(currentStep - 1);
      setResult(null);
      setRecordedURI(null);
    }
  };

  return (
    <View style={styles.container}>
      <StatusBar backgroundColor="#FF9757" barStyle="dark-content" />
      {/* Layered Curved Backgrounds */}
      <View style={styles.bgWrapper} pointerEvents="none">
        <View style={styles.bgCurve1} />
        <View style={styles.bgCurve2} />
        <View style={styles.bgCurve3} />
      </View>

      {/* Header */}
      <View style={styles.header}>
        <View style={styles.logoRow}>
          <Image
            source={require("@/assets/images/LogoHabayed.png")}
            style={{ width: 16, height: 16 }}
          />
          <Text style={styles.logoText}>Habayeb</Text>
        </View>
      </View>

      {/* Main Content */}
      <View style={styles.content}>
        <View
          style={{
            position: "absolute",
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
          }}
        >
          <Text style={styles.mainTitle}>Health Check</Text>
          <Text style={styles.subtitle}>{STEPS[currentStep].title}</Text>
          <View style={styles.stepIndicator}>
            {STEPS.map((step, index) => (
              <View
                key={step.id}
                style={[
                  styles.stepDot,
                  index === currentStep && styles.activeStepDot,
                ]}
              />
            ))}
          </View>

          {currentStep == 0 &&
            (isRecording ? (
              <Text style={styles.instruction}>
                {STEPS[currentStep].instruction}
              </Text>
            ) : (
              <Text style={styles.instruction}>
                Take a deep breath and begin
              </Text>
            ))}

          {currentStep == 1 && <DrawParkinsonSpiral />}
        </View>

        {/* {recordedURI && (
          <TouchableOpacity
            onPress={isPlaying ? stopSound : playSound}
            style={styles.playButton}
          >
            {isPlaying ? (
              <Entypo name="controller-stop" size={24} color="white" />
            ) : (
              <Entypo name="controller-play" size={24} color="white" />
            )}
          </TouchableOpacity>
        )} */}

        {/* {result && (
          <View style={styles.resultContainer}>
            <Text style={styles.resultText}>
              {result?.severity} (
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
            <Text style={styles.explanation}>{result?.explanation}</Text>
          </View>
        )} */}
      </View>

      {/* Mic Button */}
      {/* <View style={styles.micWrapper}>
        <TouchableOpacity
          style={styles.micButton}
          onPress={isRecording ? stopRecording : startRecording}
          activeOpacity={0.7}
        >
          <MaterialIcons name="keyboard-voice" size={40} color="#222" />
        </TouchableOpacity>
      </View> */}
      {currentStep == 0 && (
        <View style={styles.micWrapper}>
          <TouchableOpacity
            onPress={isRecording ? stopRecording : startRecording}
            style={[
              styles.micButton,
              !isRecording && styles.micButtonRecording,
            ]}
          >
            {loading ? (
              <ActivityIndicator size="large" color="#222" />
            ) : isRecording ? (
              <AnimatedMaterialIcons
                name="keyboard-voice"
                size={40}
                color="white"
                style={[styles.recordIcon, animatedStyles]}
              />
            ) : (
              <MaterialIcons name="keyboard-voice" size={40} color="#222" />
            )}
          </TouchableOpacity>
          {result && (
            <Text
              style={{
                color: result.prediction == 0 ? "#16A34A" : "#DC2626",
                backgroundColor: "#ffffff4C",
                padding: 10,
                borderRadius: 6,
                fontWeight: 500,
                fontSize: 18
              }}
            >
              {result.prediction == 0
                ? "You are healthy"
                : "You propably have COVID19"}
            </Text>
          )}
        </View>
      )}

      {/* Next Button */}
      <View style={styles.nextPreviousWrapper}>
        {currentStep == 1 && (
          <View
            style={{
              overflow: "hidden",
              borderRadius: 100,
              flex: 1,
              backgroundColor: "#2196F3",
            }}
          >
            <Pressable
              android_ripple={{
                color: "rgba(255, 255, 255, 0.2)",
                borderless: false,
              }}
              style={styles.nextButton}
              onPress={handlePrevious}
            >
              <Text style={styles.nextText}>Previous</Text>
            </Pressable>
          </View>
        )}
        {currentStep == 0 && (
          <View
            style={{
              overflow: "hidden",
              borderRadius: 100,
              flex: 1,
              backgroundColor: "#2196F3",
            }}
          >
            <Pressable
              android_ripple={{
                color: "rgba(254, 254, 254, 0.2)",
                borderless: false,
              }}
              style={styles.nextButton}
              onPress={handleNext}
              disabled={currentStep === STEPS.length - 1}
            >
              <Text style={styles.nextText}>Next</Text>
            </Pressable>
          </View>
        )}
      </View>
    </View>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: "#FF9757",
  },
  bgWrapper: {
    ...StyleSheet.absoluteFillObject,
    zIndex: 0,
  },
  drawerWrapper: {
    ...StyleSheet.absoluteFillObject,
    zIndex: 0,
    backgroundColor: "#CEFFF9",
  },
  bgCurve1: {
    position: "absolute",
    top: "8%",
    aspectRatio: 1,
    left: "-100%",
    width: "300%",
    borderRadius: 1000000,
    backgroundColor: "#FDC55F",
  },
  bgCurve2: {
    position: "absolute",
    top: "30%",
    aspectRatio: 1,
    left: "-100%",
    width: "300%",
    borderRadius: 1000000,
    backgroundColor: "#F4E788",
  },
  bgCurve3: {
    position: "absolute",
    top: "55%",
    aspectRatio: 1,
    left: "-100%",
    width: "300%",
    borderRadius: 1000000,
    backgroundColor: "#E27633",
  },
  drawerCurve1: {
    position: "absolute",
    top: "50%",
    aspectRatio: 1,
    left: "-50%",
    width: "300%",
    borderRadius: 1000000,
    backgroundColor: "#72E264",
  },
  drawerCurve2: {
    position: "absolute",
    top: "50%",
    aspectRatio: 1,
    left: "-150%",
    width: "300%",
    borderRadius: 1000000,
    backgroundColor: "#C6F290",
  },
  drawerCurve3: {
    position: "absolute",
    top: "60%",
    aspectRatio: 1,
    left: "-80%",
    width: "300%",
    borderRadius: 1000000,
    backgroundColor: "#3DC456",
  },
  drawerCurve4: {
    position: "absolute",
    top: "66%",
    aspectRatio: 1,
    left: "-120%",
    width: "300%",
    borderRadius: 1000000,
    backgroundColor: "#129F2C",
  },
  header: {
    display: "flex",
    flexDirection: "row",
    justifyContent: "space-between",
    alignItems: "center",
    marginTop: 12,
    marginLeft: 24,
    zIndex: 2,
  },
  logoRow: {
    flexDirection: "row",
    alignItems: "center",
    gap: 12,
  },
  logoCircle: {
    width: 20,
    height: 20,
    borderRadius: 10,
    backgroundColor: "#4A90E2",
    marginRight: 8,
    borderWidth: 3,
    borderColor: "#E3E8F0",
  },
  logoText: {
    fontSize: 20,
    fontWeight: "700",
    color: "#222",
    letterSpacing: 0.5,
  },
  content: {
    alignItems: "center",
    fontFamily: "Poppins-Regular",
    marginTop: 60,
    zIndex: 2,
  },
  mainTitle: {
    fontSize: 36,
    fontWeight: "800",
    fontFamily: "Poppins-Italic",
    color: "#222",
    marginBottom: 8,
    letterSpacing: 0.2,
  },
  subtitle: {
    fontSize: 22,
    fontWeight: "600",
    color: "#222",
    marginBottom: 8,
  },
  stepIndicator: {
    flexDirection: "row",
    alignItems: "center",
    justifyContent: "center",
    gap: 10,
    marginBottom: 30,
  },
  stepDot: {
    width: 8,
    height: 8,
    borderRadius: 4,
    backgroundColor: "rgba(0, 0, 0, 0.9)",
  },
  activeStepDot: {
    backgroundColor: "blue",
    width: 12,
    height: 12,
    borderRadius: 6,
  },
  instruction: {
    fontSize: 18,
    marginTop: 28,
    fontWeight: 600,
    textAlign: "center",
    marginBottom: 30,
  },
  recordButton: {
    marginBottom: 20,
  },
  recordingIcon: {
    backgroundColor: "#E74C3C",
    padding: 20,
    borderRadius: 50,
  },
  recordIcon: {
    padding: 14,
    backgroundColor: "rgb(226, 30, 30)",
    borderRadius: 50,
  },
  playButton: {
    padding: 16,
    backgroundColor: "rgba(255, 255, 255, 0.2)",
    borderRadius: 50,
    marginBottom: 20,
  },
  resultContainer: {
    width: "100%",
    paddingHorizontal: 20,
    marginTop: 20,
  },
  resultText: {
    fontSize: 18,
    fontWeight: "600",
    textAlign: "center",
    color: "white",
  },
  progressBarBackground: {
    width: "100%",
    height: 8,
    backgroundColor: "rgba(255, 255, 255, 0.2)",
    borderRadius: 4,
    marginTop: 10,
    overflow: "hidden",
  },
  progressBarFill: {
    height: "100%",
    borderRadius: 4,
  },
  explanation: {
    paddingVertical: 10,
    fontSize: 16,
    padding: 10,
    backgroundColor: "rgba(255, 255, 255, 0.1)",
    marginTop: 10,
    color: "white",
    borderRadius: 8,
  },
  micWrapper: {
    flex: 1,
    position: "absolute",
    left: "50%",
    gap: 20,
    bottom: 150,
    justifyContent: "center",
    alignItems: "center",
    zIndex: 2,
    transform: [{ translateX: "-50%" }], // Using transform to move left by 50%
  },
  micButtonRecording: {
    backgroundColor: "#8EC6FF",
    shadowColor: "#000",
    shadowOpacity: 0.08,
    shadowRadius: 8,
    shadowOffset: { width: 0, height: 2 },
    elevation: 4,
  },
  micButton: {
    width: 76,
    height: 76,
    borderRadius: 45,
    justifyContent: "center",
    alignItems: "center",
  },
  nextWrapper: {
    width: "100%",
    alignItems: "center",
    marginBottom: 32,
    zIndex: 2,
  },
  nextPreviousWrapper: {
    position: "absolute",
    bottom: 16,
    width: "100%",
    maxHeight: 64,
    display: "flex",
    flexDirection: "row",
    alignItems: "center",
    justifyContent: "center",
    gap: 8,
    paddingHorizontal: 8,
    zIndex: 2,
  },
  nextButton: {
    flex: 1,
    borderRadius: 32,
    paddingVertical: 10,
    alignItems: "center",
    justifyContent: "center",
  },
  nextText: {
    color: "#fff",
    fontSize: 20,
    fontWeight: "700",
    letterSpacing: 0.5,
  },
  menuButton: {
    marginRight: 16,
  },
  drawer: {
    width: screenWidth * 0.75,
  },
  drawerContainer: {
    flex: 1,
    padding: 20,
    backgroundColor: "#fff",
    overflow: "hidden",
  },
  drawerHeader: {
    flexDirection: "row",
    alignItems: "center",
    paddingBottom: 20,
    borderBottomWidth: 1,
    borderBottomColor: "#eee",
    marginBottom: 20,
  },
  drawerTitle: {
    fontSize: 22,
    fontWeight: "700",
    marginLeft: 12,
    color: "#222",
  },
  drawerItems: {},
  drawerItem: {
    flexDirection: "row",
    alignItems: "center",
    paddingVertical: 15,
    paddingHorizontal: 10,
    borderRadius: 8,
    marginBottom: 5,
  },
  drawerItemText: {
    fontSize: 18,
    fontWeight: 600,
    marginLeft: 15,
    color: "#333",
  },
});

export default RecordingButton;
