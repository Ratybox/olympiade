import React, { useEffect, useState } from "react";
import {
  StyleSheet,
  View,
  Dimensions,
  Alert,
  ActivityIndicator,
  Text,
} from "react-native";
import MapView, { UrlTile, Marker, Region } from "react-native-maps";
import axios from "axios";
import * as Location from "expo-location";

export default function DoctorMap() {
  const [doctors, setDoctors] = useState<any[]>([]);
  const [region, setRegion] = useState<Region | null>(null);

  useEffect(() => {
    getCurrentLocation();
  }, []);

  async function getCurrentLocation() {
    const { status } = await Location.requestForegroundPermissionsAsync();
    if (status !== "granted") {
      Alert.alert(
        "Permission Denied",
        "Location access is required to show doctors nearby."
      );
      return;
    }

    const location = await Location.getCurrentPositionAsync({});
    const { latitude, longitude } = location.coords;

    setRegion({
      latitude,
      longitude,
      latitudeDelta: 0.1,
      longitudeDelta: 0.1,
    });

    console.info(latitude, longitude);

    fetchDoctors(latitude, longitude);
  }

  async function fetchDoctors(lat: number, lon: number) {
    const delta = 0.2; // adjust search radius (~degrees)

    try {
      const response = await axios.get(
        "https://nominatim.openstreetmap.org/search",
        {
          params: {
            q: "doctor",
            format: "json",
            limit: 20,
            bounded: 1, // strict bounding
            viewbox: [
              lon - delta, // left
              lat + delta, // top
              lon + delta, // right
              lat - delta, // bottom
            ].join(","),
          },
          headers: {
            "User-Agent": "HabayebApp/1.0 (contact@habayeb.com)",
          },
        }
      );

      console.info("fetched doctors: ", response.data);
      setDoctors(response.data);
    } catch (err) {
      console.error("Error fetching doctors:", err);
    }
  }

  return (
    <View style={styles.container}>
      {!region ? (
        <View style={styles.loadingContainer}>
          <ActivityIndicator size="large" color="#0000ff" />
          <Text style={styles.loadingText}>Searching for the nearest doctors in your area</Text>
        </View>
      ) : (
        <MapView style={styles.map} initialRegion={region} provider={undefined}>
          <UrlTile
            urlTemplate="http://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
            maximumZ={19}
            flipY={false}
          />

          {/* Your current location marker */}
          <Marker
            coordinate={{
              latitude: region.latitude,
              longitude: region.longitude,
            }}
            pinColor="blue"
            title="You are here"
          />

          {doctors.map((doc: any, index: number) => (
            <Marker
              key={index}
              coordinate={{
                latitude: parseFloat(doc.lat),
                longitude: parseFloat(doc.lon),
              }}
              title={doc.display_name.split(",")[0]}
              description={doc.display_name}
            />
          ))}
        </MapView>
      )}
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
  },
  loadingContainer: {
    backgroundColor: "white",
    flex: 1,
    gap: 20,
    display: "flex",
    alignItems: "center",
    justifyContent: "center",
  },
  map: {
    width: Dimensions.get("window").width,
    height: Dimensions.get("window").height,
  },
  loadingText: {
    textAlign: "center",
    fontWeight: 500,
    fontSize: 16,
  },
});
