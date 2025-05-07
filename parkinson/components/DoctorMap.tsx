import React from 'react';
import { StyleSheet, View, Dimensions } from 'react-native';
import MapView, { UrlTile } from 'react-native-maps';

export default function DoctorMap() {
  return (
    <View style={styles.container}>
      <MapView
        style={styles.map}
        initialRegion={{
          latitude: 36.75,
          longitude: 3.06,
          latitudeDelta: 0.05,
          longitudeDelta: 0.05,
        }}
        provider={undefined} // Important: use null for custom tiles like OSM
      >
        <UrlTile
          urlTemplate="http://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
          maximumZ={19}
          flipY={false}
        />
      </MapView>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
  },
  map: {
    width: Dimensions.get('window').width,
    height: Dimensions.get('window').height,
  },
});
