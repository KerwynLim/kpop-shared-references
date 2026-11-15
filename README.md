# K-pop shared references

Reference faces shared by fans for **Sort by Member**, so everyone's app recognises members
without collecting reference photos one by one.

## Use them
In Sort by Member: **Settings → Shared references → Online → Check for Shared References…**,
tick the packs you want, **Add Selected**. Adding only adds — your own groups, members and
reference photos are never changed or removed. Offline (the default) never connects.

## Contribute
1. Sort by Member: **Settings → Shared references → Contribute Your References…** — pick the
   groups, save the `.zip`.
2. [Open a contribution](https://github.com/KerwynLim/kpop-shared-references/issues/new?template=contribute.yml),
   drag the `.zip` in, submit.

A robot checks it within minutes and adds it for everyone. It keeps only valid files and
names, and only pictures that contain the labelled face; it removes everything else, including
the contributor's face numbers (each app computes its own from the pictures). **It can't tell
whether a face is labelled with the right member — please label carefully.**

Everything here is public. Share only face crops you're happy for anyone to see.

## Layout
* `index.json` — the list the app reads
* `packs/*.sbmpack` — approved packs (zip: `manifest.json` + `refs/*.jpg`)
* `tools/check_and_publish.py` + `.github/workflows/auto-approve.yml` — the automatic check
* `tools/model/` — YuNet face detector (MIT licence, OpenCV Zoo)
