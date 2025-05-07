import {
  View,
  Image,
  Text,
  TouchableOpacity,
  StyleSheet,
  Dimensions,
} from "react-native";
import { useState } from "react";
import { Drawer } from "react-native-drawer-layout";
import { Slot, useRouter } from "expo-router";
import { useColorScheme } from "react-native";
import {
  ThemeProvider,
  DarkTheme,
  DefaultTheme,
} from "@react-navigation/native";
import {
  Entypo,
  MaterialIcons,
  Ionicons,
  FontAwesome5,
} from "@expo/vector-icons";

const { width: screenWidth, height: screenHeight } = Dimensions.get("window");

export default function DrawerLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  const [open, setOpen] = useState(false);
  const colorScheme = useColorScheme();
  const router = useRouter();

  const toggleDrawer = () => {
    setOpen(!open);
  };

  const renderDrawerContent = () => {
    return (
      <View style={styles.drawerContainer}>
        <View style={styles.drawerWrapper} pointerEvents="none">
          <View style={styles.drawerCurve1} />
          <View style={styles.drawerCurve2} />
          <View style={styles.drawerCurve3} />
          <View style={styles.drawerCurve4} />
        </View>

        <View style={styles.drawerHeader}>
          <Image
            source={require("@/assets/images/LogoHabayed.png")}
            style={{ width: 24, height: 24 }}
          />
          <Text style={styles.drawerTitle}>Habayeb</Text>
        </View>

        <View style={styles.drawerItems}>
          <TouchableOpacity
            style={styles.drawerItem}
            onPress={() => {
              setOpen(false);
              router.push("/habayeb")
            }}
          >
            <Image
              source={require("@/assets/images/Habayeb.png")}
              style={{ width: 24, height: 24 }}
            />
            <Text style={styles.drawerItemText}>Habayeb</Text>
          </TouchableOpacity>

          <TouchableOpacity
            style={styles.drawerItem}
            onPress={() => {
              setOpen(false);
              router.push("/");
            }}
          >
            <FontAwesome5 name="heartbeat" size={24} color="#333" />
            <Text style={styles.drawerItemText}>Health Check</Text>
          </TouchableOpacity>

          <TouchableOpacity
            style={styles.drawerItem}
            onPress={() => {
              setOpen(false);
              // navigation.navigate('History');
            }}
          >
            <Ionicons name="time-outline" size={24} color="#333" />
            <Text style={styles.drawerItemText}>History</Text>
          </TouchableOpacity>

          <TouchableOpacity
            style={styles.drawerItem}
            onPress={() => {
              setOpen(false);
              router.push("/maps");
            }}
          >
            <Ionicons name="map-outline" size={24} color="#333" />
            <Text style={styles.drawerItemText}>Map</Text>
          </TouchableOpacity>

          <TouchableOpacity
            style={styles.drawerItem}
            onPress={() => {
              setOpen(false);
              // navigation.navigate('Settings');
            }}
          >
            <Ionicons name="settings-outline" size={24} color="#333" />
            <Text style={styles.drawerItemText}>Settings</Text>
          </TouchableOpacity>
        </View>

        {/* <Image
          source={require("@/assets/images/Habayeb_drawer.png")}
          style={{
            width: 120,
            height: 120,
            position: "absolute",
            bottom: 22,
            left: "70%",
            transform: [{ translateX: "-40%" }],
          }}
        /> */}
      </View>
    );
  };

  return (
    <ThemeProvider value={colorScheme === "dark" ? DarkTheme : DefaultTheme}>
      <Drawer
        open={open}
        onOpen={() => setOpen(true)}
        onClose={() => setOpen(false)}
        drawerPosition="right"
        renderDrawerContent={renderDrawerContent}
        drawerStyle={styles.drawer}
      >
        <TouchableOpacity onPress={toggleDrawer} style={styles.menuButton}>
          <Ionicons name="menu" size={28} color="#222" />
        </TouchableOpacity>
        {children}
      </Drawer>
    </ThemeProvider>
  );
}

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
  recordingInstruction: {
    fontSize: 20,
    textAlign: "center",
    marginBottom: 30,
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
    backgroundColor: "rgba(255, 255, 255, 1)",
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
    bottom: 220,
    justifyContent: "center",
    alignItems: "center",
    zIndex: 2,
    transform: [{ translateX: "-50%" }], // Using transform to move left by 50%
  },
  micButton: {
    width: 76,
    height: 76,
    borderRadius: 45,
    backgroundColor: "#8EC6FF",
    justifyContent: "center",
    alignItems: "center",
    shadowColor: "#000",
    shadowOpacity: 0.08,
    shadowRadius: 8,
    shadowOffset: { width: 0, height: 2 },
    elevation: 4,
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
    position: "absolute",
    top: 12,
    right: 8,
    marginRight: 16,
    zIndex: 50,
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
