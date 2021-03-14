# Mobile Debugging

## Firefox + Android

On the phone

* Enable "USB debugging" in system->developer options
* Play with the USB mode in "Connected devices". Mine works with "PTP" (picture transfer protocol), but that doesn't really make sense...

On the computer

* Install adb
* Run `adb devices` and ensure you see the phone
* Go to `about:debugging` in Firefox and hopefully the device will be on the left
* Click on the device. You can then inspect its tabs (basically get the dev tools for that page)

See [mozilla docs](https://developer.mozilla.org/en-US/docs/Tools/about:debugging).

## Chrome + Android

On the phone -- pretty much the same as Firefox.

On the computer -- also pretty much the same, until we get to the Firefox specific stuff.

* Go to `chrome://inspect/#devices`. Click inspect on the device and tab you care about!

See [chrome docs](https://developers.google.com/web/tools/chrome-devtools/remote-debugging).
Interestingly, these suggest setting the USB mode to PTP!
