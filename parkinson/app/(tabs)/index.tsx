import React, { useState, useEffect } from "react";
import { View, TouchableOpacity, Text, StyleSheet } from "react-native";
import { Audio } from "expo-av";
import { Recording, Sound } from "expo-av/build/Audio";
import { Entypo, MaterialIcons } from "@expo/vector-icons";
import * as FileSystem from "expo-file-system";
import axios from "@/utils/axios";

const RecordingButton = () => {
  const [recording, setRecording] = useState<null | Recording>(null);
  const [isRecording, setIsRecording] = useState(false);
  const [recordingStatus, setRecordingStatus] = useState("Ready to record");
  const [recordedURI, setRecordedURI] = useState<null | string>(null);
  const [sound, setSound] = useState<null | Sound>(null);
  const [isPlaying, setIsPlaying] = useState(false);

  async function startRecording() {
    try {
      console.log("Requesting permissions..");
      await Audio.requestPermissionsAsync();
      await Audio.setAudioModeAsync({
        allowsRecordingIOS: true,
        playsInSilentModeIOS: true,
      });

      console.log("Starting recording..");
      const { recording } = await Audio.Recording.createAsync(
        Audio.RecordingOptionsPresets.HIGH_QUALITY
      );

      setRecording(recording);
      setIsRecording(true);
      setRecordingStatus("Recording in progress...");
      console.log("Recording started");
    } catch (err) {
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

      const response = await axios.post("/recordings", formData, {
        headers: {
          "Content-Type": "multipart/form-data",
        },
        timeout: 5000,
      });

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
        uploadAudio(uri!)
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

  // Clean up sound on unmount
  useEffect(() => {
    return sound
      ? () => {
          console.log("Unloading Sound");
          sound.unloadAsync();
        }
      : undefined;
  }, [sound]);

  return (
    <View style={styles.container}>
      <Text
        style={{ position: "absolute", top: 20, fontSize: 24, fontWeight: 700 }}
      >
        Parkinson Disease Detector
      </Text>
      <TouchableOpacity onPress={isRecording ? stopRecording : startRecording}>
        {isRecording ? (
          <MaterialIcons
            name="keyboard-voice"
            size={40}
            color="white"
            style={{ backgroundColor: "red", padding: 16, borderRadius: 50 }}
          />
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

      <Text style={styles.statusText}>{recordingStatus}</Text>
    </View>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
    justifyContent: "center",
    alignItems: "center",
    gap: 20,
    backgroundColor: "white",
    padding: 20,
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
});

export default RecordingButton;
