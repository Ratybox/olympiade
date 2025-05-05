import React, { useState, useRef } from "react";
import {
  View,
  StyleSheet,
  Dimensions,
  TouchableOpacity,
  ActivityIndicator,
  Text,
  Pressable,
} from "react-native";
import { Svg, Path } from "react-native-svg";
import { captureRef } from "react-native-view-shot";
import { Image } from "react-native";
import * as ImageManipulator from "expo-image-manipulator";
import axios from "@/utils/axios";
import { MaterialIcons } from "@expo/vector-icons";

const { height, width } = Dimensions.get("window");
const svg_dim = width * 0.9;

export default () => {
  const [paths, setPaths] = useState<any>([]);
  const [currentPath, setCurrentPath] = useState<any>([]);
  const [isClearButtonClicked, setClearButtonClicked] = useState(false);
  const svgContainerRef = useRef(null);
  const [imageUri, setImageUri] = useState<any>(null);
  const [result, setResult] = useState<string>("Nothing");
  const [loading, setLoading] = useState<boolean>(false); // State for loading
  const [liked, setLiked] = useState<boolean>(false);

  const handleSubmitButtonClick = async () => {
    setLoading(true);
    try {
      // Capture the SVG container as an image
      const uri = await captureRef(svgContainerRef, {
        format: "png",
        quality: 1,
      });
      setImageUri(uri); // Save URI to state

      // Resize the image to 28x28 pixels
      const resizedImage = await ImageManipulator.manipulateAsync(
        uri,
        [{ resize: { width: 28, height: 28 } }],
        { base64: true } // Get the base64 representation
      );

      // Extract the pixel matrix (optional, depends on your use case)
      const base64Image = resizedImage.base64;
      console.info("28x28 Image Base64:", base64Image);

      axios
        .post("pca_digits/", { image: base64Image })
        .then((res) => {
          setResult(res.data.message);
          console.info(res.data.message);
          setLoading(false);
        })
        .catch((err) => {
          console.error("Error while posting image: ", err);
          setLoading(false);
        });
      // Pass the base64 or pixel matrix to your backend or further processing
    } catch (error) {
      console.error("Error capturing drawing:", error);
    }
  };

  const onTouchEnd = () => {
    paths.push(currentPath);
    setCurrentPath([]);
    setClearButtonClicked(false);
  };

  const onTouchMove = (event: any) => {
    const newPath = [...currentPath];
    const locationX = event.nativeEvent.locationX;
    const locationY = event.nativeEvent.locationY;
    const newPoint = `${newPath.length === 0 ? "M" : ""}${locationX.toFixed(
      0
    )},${locationY.toFixed(0)} `;
    newPath.push(newPoint);
    setCurrentPath(newPath);
  };

  const handleClearButtonClick = () => {
    setPaths([]);
    setCurrentPath([]);
    setClearButtonClicked(true);
  };

  const handleLike = () => {
    setLiked((prev) => !prev);
  };

  return (
    <View style={{ display: "flex", flexDirection: "column", gap: 10 }}>
      <View
        style={styles.svgContainer}
        onTouchMove={onTouchMove}
        onTouchEnd={onTouchEnd}
      >
        <Svg
          ref={svgContainerRef}
          height={svg_dim}
          width={svg_dim}
          className="rounded-2xl"
        >
          {/* {paths.length == 0 && currentPath.length == 0 && <OneDigitIcon />} */}
          {paths.length == 0 && currentPath == 0 && (
            <Text
              style={{
                position: "absolute",
                top: 10,
                left: "50%",
                transform: [{ translateX: "-50%" }],
              }}
            >
              Draw a spiral inside this screen
            </Text>
          )}
          <Path
            d={paths.join("")}
            stroke={"gray"}
            fill={"transparent"}
            strokeWidth={20}
            strokeLinejoin={"round"}
            strokeLinecap={"round"}
          />
          <Path
            d={currentPath.join("")}
            stroke={"gray"}
            fill={"transparent"}
            strokeWidth={15}
            strokeLinejoin={"round"}
            strokeLinecap={"round"}
          />
        </Svg>
      </View>
      <View
        style={{
          flexDirection: "row",
          alignItems: "center",
          justifyContent: "space-between",
          gap: 12,
          width: svg_dim,
          height: "auto",
        }}
      >
        <View
          style={{
            overflow: "hidden",
            borderRadius: 100,
            backgroundColor: "#2196F3",
          }}
        >
          <Pressable
            android_ripple={{
              color: "rgba(255, 255, 255, 0.2)",
              borderless: false,
            }}
            style={{ padding: 12 }}
            onPress={handleClearButtonClick}
          >
            <MaterialIcons name="delete" size={24} color="white" />
          </Pressable>
        </View>
        <View
          style={{
            overflow: "hidden",
            borderRadius: 100,
            backgroundColor: "#2196F3",
          }}
        >
          <Pressable
            android_ripple={{
              color: "rgba(255, 255, 255, 0.2)",
              borderless: false,
            }}
            style={{ padding: 12 }}
            onPress={handleSubmitButtonClick}
          >
            <MaterialIcons name="check" size={24} color="white" />
          </Pressable>
        </View>
      </View>
      {/* <View className="min-w-full h-20 bg-[#27AF84E6] rounded-full flex flex-row items-center justify-between p-[5px]">
        <View className="w-fit h-full flex items-center justify-center px-6 bg-[#FFFFFF1C] rounded-full">
          <Text className="text-white text-base font-medium">
            Predicted input: {result}
          </Text>
        </View>
        <Button onPress={handleLike} classNameStyle="bg-transparent">
          {!liked ? <LikeIcon /> : <LikeIconActive />}
        </Button>
      </View> */}
      {/* loading && (
        <ActivityIndicator
          size="large"
          color="blue"
          style={styles.loadingIndicator}
        />
      )
      imageUri && (
          <>
            <Image
              source={{ uri: imageUri }}
              style={styles.capturedImage}
              resizeMode="contain"
            />
          </>
        )
      <Text className="text-4xl bg-black/10 py-2 px-2 rounded-md font-medium">
        You have entered: {result}
      </Text> */}
    </View>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
    justifyContent: "center",
    alignItems: "center",
  },
  svgContainer: {
    height: svg_dim,
    width: svg_dim,
    backgroundColor: "rgba(255, 255, 255, 0.6)",
    borderRadius: 40,
    backdropFilter: "blur(15px)",
  },
  clearButton: {
    marginTop: 10,
    backgroundColor: "black",
    paddingVertical: 10,
    paddingHorizontal: 20,
    borderRadius: 5,
  },
  clearButtonText: {
    color: "white",
    fontSize: 16,
    fontWeight: "bold",
  },
  capturedImage: {
    marginTop: 20,
    width: 100,
    height: 100,
    borderColor: "black",
    borderWidth: 1,
  },
  loadingIndicator: {
    marginTop: 20,
  },
});
