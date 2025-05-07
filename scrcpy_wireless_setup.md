# 📲 Scrcpy: Wireless Phone Mirroring Setup

## ✅ Prerequisites
- USB Debugging must be enabled on your Android device.
- Your **PC and Android device must be on the same Wi-Fi network**.
- `adb` and `scrcpy` should be installed on your PC.

---

## 🔌 Step 1: Connect Your Phone via USB (Initial Setup)

Run the following commands in a terminal (CMD or PowerShell):

```bash
adb devices               # Confirm your phone is detected
adb shell ip route        # Find your phone's local IP address
adb tcpip 5555            # Enable wireless debugging on port 5555
```

---

## 📡 Step 2: Connect via Wi-Fi

Disconnect the USB cable, then run this command, replacing `<your_phone_ip>` with the IP address from the previous step:

```bash
adb connect <your_phone_ip>:5555
```

Example:

```bash
adb connect 192.168.46.21:5555
```

You should see a message like:

```
connected to 192.168.46.21:5555
```

---

## 🖥️ Step 3: Launch Scrcpy Wirelessly

To start mirroring your device screen:

```bash
scrcpy
```

If you're using the Windows `.exe` version, you can also double-click `scrcpy.exe` to launch it.

✅ Your Android screen should now appear on your PC!