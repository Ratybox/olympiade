import { Recording, Sound } from "expo-av/build/Audio";
import React, { useState } from "react";
import {
  StatusBar,
  StyleSheet,
  Text,
  TouchableOpacity,
  View,
} from "react-native";
import Animated, {
  useSharedValue,
  useAnimatedStyle,
  withRepeat,
  withTiming,
  Easing,
  interpolate,
  withSpring,
} from "react-native-reanimated";
import { Audio } from "expo-av";
import axios from "@/utils/axios";
import * as FileSystem from "expo-file-system";
import { MaterialIcons } from "@expo/vector-icons";

const AnimatedMaterialIcons = Animated.createAnimatedComponent(MaterialIcons);


const Habayeb = () => {
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
  const [transcription, setTranscription] = useState('');
  const [responseText, setResponseText] = useState('');
  const [isProcessing, setIsProcessing] = useState(false);

  const scale = useSharedValue(1);
  const opacity = useSharedValue(1);

  const animatedStyles = useAnimatedStyle(() => {
    return {
      transform: [{ scale: scale.value }],
      opacity: opacity.value,
    };
  });

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
      });

      setResult(response.data);

      console.log("Upload result:", response.data);
      setRecordingStatus("Upload successful!");
    } catch (error: any) {
      console.error("Upload error:", error);
      setRecordingStatus("Upload failed: " + error.message);
    }
  }

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
        processAudio(uri!);
        // setRecordingStatus("Recording stopped and stored at: " + uri);
        // console.log("Recording stopped and stored at", uri);
      }
    } catch (err) {
      console.error("Failed to stop recording", err);
      setRecordingStatus("Failed to stop recording");
    }
  }

   const processAudio = async (uri: string) => {
    setIsProcessing(true);
    try {
      // Read the file into a binary format
      const fileInfo = await FileSystem.getInfoAsync(uri);
      const fileUri = fileInfo.uri;
      const fileType = 'audio/m4a';

      const formData = new FormData();
      formData.append('file', {
        uri: fileUri,
        name: 'audio.m4a',
        type: fileType,
      } as any);
      formData.append('model', 'whisper-1');

      // Transcribe audio using OpenAI Whisper API
      const transcriptionResponse = await axios.post(
        'https://api.openai.com/v1/audio/transcriptions',
        formData,
        {
          headers: {
            'Content-Type': 'multipart/form-data',
            Authorization: `Bearer ${OPENAI_API_KEY}`,
          },
        }
      );

      const transcribedText = transcriptionResponse.data.text;
      setTranscription(transcribedText);
      console.log('Transcription:', transcribedText);

      // Send transcription to ChatGPT
      const chatResponse = await axios.post(
        'https://api.openai.com/v1/chat/completions',
        {
          model: 'gpt-4',
          messages: [
            {
              role: 'system',
              content: 'You are a compassionate psychiatrist helping a user with mental wellbeing.',
            },
            { role: 'user', content: transcribedText },
          ],
        },
        {
          headers: {
            'Content-Type': 'application/json',
            Authorization: `Bearer ${OPENAI_API_KEY}`,
          },
        }
      );

      const assistantMessage = chatResponse.data.choices[0].message.content;
      setResponseText(assistantMessage);
      console.log('Assistant Response:', assistantMessage);

      // Convert assistant message to speech using OpenAI TTS
      const ttsResponse = await axios.post(
        'https://api.openai.com/v1/audio/speech',
        {
          model: 'tts-1',
          input: assistantMessage,
          voice: 'nova',
        },
        {
          headers: {
            'Content-Type': 'application/json',
            Authorization: `Bearer ${OPENAI_API_KEY}`,
          },
          responseType: 'arraybuffer',
        }
      );

      // Save the audio file
      const ttsUri = FileSystem.documentDirectory + 'response.mp3';
      await FileSystem.writeAsStringAsync(ttsUri, Buffer.from(ttsResponse.data).toString('base64'), {
        encoding: FileSystem.EncodingType.Base64,
      });

      // Play the audio
      const { sound } = await Audio.Sound.createAsync({ uri: ttsUri });
      await sound.playAsync();
    } catch (error) {
      console.error('Error processing audio:', error);
    } finally {
      setIsProcessing(false);
    }
  };


  return (
    <View style={styles.container}>
      <StatusBar backgroundColor="#2DB1EA" barStyle="dark-content" />
      <View style={styles.bgWrapper}>
        <View
          style={[
            styles.circle,
            styles.circle1,
            { backgroundColor: "#E5F7FD", zIndex: 4 },
          ]}
        />
        <View
          style={[
            styles.circle,
            styles.circle2,
            { backgroundColor: "#90E3F2", zIndex: 3 },
          ]}
        />
        <View
          style={[
            styles.circle,
            styles.circle3,
            { backgroundColor: "#70D9EB", zIndex: 2 },
          ]}
        />
        <View
          style={[
            styles.circle,
            styles.circle4,
            { backgroundColor: "#49C3F8", zIndex: 1 },
          ]}
        />
      </View>

      <View style={styles.micWrapper}>
        <TouchableOpacity
          onPress={isRecording ? stopRecording : startRecording}
          style={[styles.micButton, !isRecording && styles.micButtonRecording]}
        >
          {isRecording ? (
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
      </View>

      <View style={styles.textContainer}>
        <Text style={styles.mainTitle}>How are you feeling today?</Text>
        <Text style={styles.subtitle}>
          Tap the mic and speak a few words or make a vocal sound.
        </Text>

        {/* <Text style={styles.status}>{recordingStatus}</Text> */}

        {/* {result && (
          <Text style={styles.result}>
            {result.predicted_class === 1
              ? "Signs of emotional stress detected"
              : "You sound emotionally stable"}
            {"\n"}Confidence: {(result.confidence * 100).toFixed(1)}%
          </Text>
        )} */}
      </View>
    </View>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: "#2DB1EA",
    justifyContent: "center",
    alignItems: "center",
  },
  bgWrapper: {
    justifyContent: "center",
    alignItems: "center",
  },
  circle: {
    position: "absolute",
    borderRadius: 10000,
  },
  circle1: {
    width: "148%",
    aspectRatio: 1,
  },
  circle2: {
    width: "160%",
    aspectRatio: 1,
  },
  circle3: {
    width: "172%",
    aspectRatio: 1,
  },
  circle4: {
    width: "184%",
    aspectRatio: 1,
  },
  micWrapper: {
    flex: 1,
    position: "absolute",
    justifyContent: "center",
    alignItems: "center",
    zIndex: 5,
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
  recordIcon: {
    padding: 14,
    backgroundColor: "rgb(219, 58, 58)",
    borderRadius: 50,
  },

  textContainer: {
    position: "absolute",
    top: "25%",
    paddingHorizontal: 24,
    alignItems: "center",
    zIndex: 6,
  },

  mainTitle: {
    fontSize: 26,
    fontWeight: "700",
    textAlign: "center",
    marginBottom: 10,
  },

  subtitle: {
    fontSize: 16,
    textAlign: "center",
    marginBottom: 20,
  },

  status: {
    fontSize: 14,
    fontStyle: "italic",
    textAlign: "center",
  },

  result: {
    fontSize: 16,
    fontWeight: "600",
    marginTop: 20,
    textAlign: "center",
    backgroundColor: "rgba(0, 0, 0, 0.2)",
    padding: 12,
    borderRadius: 12,
  },
});

export default Habayeb;
