---
title: My first Arch Linux install
date: 2020-04-15
description: "My experiences with installing Arch linux for the first time"
tags:
  - linux
---

I recently installed arch linux on a HP Pavilion 15-au109ne, which my roommate gave to me because Windows had gone bonkers on it(surprise, surprise). This was quite the ordeal, because of the `Realtek RTL8723BE` wifi card/controller it had, which caused a lot of issues.

Neither Pop!\_OS or Ubuntu (both 19.10) worked quite correctly, even after the fix required to work with this card(which was adding a kernel parameter, linked below). In the end, I decided to go with Arch because :

- Up to date and bleeding edge packages(Including the newest gnome 3.36, which even my laptop doesn’t have). I thought this would help make the laptop actually workable.
- The arch installation process terrified me, and I wanted to learn about it to gain a better understanding of my computer.
- Being able to quote “I use arch btw”

Anyways, these are the links that helped me in the process.

- The [official Arch install guide](https://wiki.archlinux.org/index.php/installation_guide#Configure_the_system), that I referred alongside [this](https://itsfoss.com/install-arch-linux/) article on It’s FOSS.
- Referred to [this video](https://www.youtube.com/watch?v=1Aup0im7QSM) for learning what mirrors are and how to set them in the arch install process.
- How to check if user is [sudo or not](https://unix.stackexchange.com/a/50787).
- Adding yourself to [sudoer group](https://bbs.archlinux.org/viewtopic.php?id=133625). Referred to [this](https://bbs.archlinux.org/viewtopic.php?id=59846) link too.
- [What to do](https://askubuntu.com/a/546090) if your `home/username` does not exist.
- How to [check for boot errors](https://bbs.archlinux.org/viewtopic.php?pid=1863992#p1863992) using `journalctl`
- GNOME terminal not launching [reason](https://unix.stackexchange.com/a/171539). Fix in [archwiki](https://wiki.archlinux.org/index.php/Locale#Gnome-terminal_or_rxvt-unicode).
- [This reddit comment](https://www.reddit.com/r/archlinux/comments/7wmio5/pcie_bus_error_during_installation/du1lupc?utm_source=share&utm_medium=web2x) and [this article](https://itsfoss.com/pcie-bus-error-severity-corrected/) which I used to fix the `PCIE bus error` the wifi card was causing. This was done once arch was installed by changing GRUB kernel parameters.

Extra:

- Installing [blackarch packages](https://blackarch.org/downloads.html) on top of an existing arch install.

I’m sure you’re all aware that the arch fandom is filled with people who hate microsoft even more than I do. I came across a really HILARIOUS quote in some person’s bio on ArchWiki:

> The day Microsoft makes a product that doesn’t suck, is the day they make a vacuum cleaner.

Have a good one folks :D
