## hordelib Changelog

## [v2.11.0](https://github.com/Haidra-Org/hordelib/compare/v2.10.0...v2.11.0)

22 May 2024

- feat: qr control support [`b1e6e56`](https://github.com/Haidra-Org/hordelib/commit/b1e6e5641e05057eca650f785a19ace735f19dec)  (db0)
- refactor: compile logging regexes in `OutputCollector` [`a43e959`](https://github.com/Haidra-Org/hordelib/commit/a43e9597a8fb54b207ced8fd0d17bb9e969cc456)  (tazlin)
- fix: catch out-of-bounds QR and missing steps [`d9f370b`](https://github.com/Haidra-Org/hordelib/commit/d9f370bd2415b6f3dbd5adf51b63e850df91d791)  (db0)

## [v2.10.0](https://github.com/Haidra-Org/hordelib/compare/v2.9.3...v2.10.0)

13 May 2024

- feat: adds stable_cascade_2pass [`#253`](https://github.com/Haidra-Org/hordelib/pull/253) (Divided by Zer0)
- feat: Removes the requirement for a design pipeline [`#251`](https://github.com/Haidra-Org/hordelib/pull/251) (Divided by Zer0)

## [v2.9.3](https://github.com/Haidra-Org/hordelib/compare/v2.9.2...v2.9.3)

5 May 2024

- ci/chore: direct link to changelog on webhook [`8ae6a26`](https://github.com/Haidra-Org/hordelib/commit/8ae6a26037d00c47d327472c4d4d36481f0fefc1)  (tazlin)

## [v2.9.2](https://github.com/Haidra-Org/hordelib/compare/v2.9.1...v2.9.2)

5 May 2024

- ci/fix: `id-token: write` perm on publish workflow [`506516a`](https://github.com/Haidra-Org/hordelib/commit/506516ac0a7e18350d16bc48b0667318aada432e)  (tazlin)

## [v2.9.1](https://github.com/Haidra-Org/hordelib/compare/v2.9.0...v2.9.1)

5 May 2024

- ci/fix: don't use pypi api token [`a6b4a29`](https://github.com/Haidra-Org/hordelib/commit/a6b4a294601bef4ffab72f77348ef8d4214cf1fa)  (tazlin)

## [v2.9.0](https://github.com/Haidra-Org/hordelib/compare/v2.8.2...v2.9.0)

5 May 2024

- tests: custom models; check+dl custom testing model [`d093012`](https://github.com/Haidra-Org/hordelib/commit/d093012ccfdffde188fb27763e6b7a1ae73a64e1)  (tazlin)
- feat: allow custom models via env var [`34b348c`](https://github.com/Haidra-Org/hordelib/commit/34b348ccac06e64dd553ad2c20384ab44b1cdb36)  (tazlin)
- docs: update readme.md [`4e1f2bb`](https://github.com/Haidra-Org/hordelib/commit/4e1f2bbaa6fe5f82fb47325cb13f6e6ab641680b)  (tazlin)

## [v2.8.2](https://github.com/Haidra-Org/hordelib/compare/v2.8.1...v2.8.2)

6 April 2024

- feat: update to comfyui `a7dd82e` [`9b666cd`](https://github.com/Haidra-Org/hordelib/commit/9b666cd859cb222e5fba1bf557320b0c77d00f4d)  (tazlin)

## [v2.8.1](https://github.com/Haidra-Org/hordelib/compare/v2.8.0...v2.8.1)

27 March 2024

- fix: avoid crash on long lora names [`ab57d9c`](https://github.com/Haidra-Org/hordelib/commit/ab57d9ceee1e1de7a8be104605bce02e71a7bcff)  (db0)

## [v2.8.0](https://github.com/Haidra-Org/hordelib/compare/v2.7.6...v2.8.0)

24 March 2024

- feat: use sdk's new source image/mask/extra image handling [`6637f76`](https://github.com/Haidra-Org/hordelib/commit/6637f763f1a34b87b4475f0445914fa16c82335f)  (tazlin)
- feat: convert extra source images to pil [`579a071`](https://github.com/Haidra-Org/hordelib/commit/579a07139f3ebf5a84adad8fa06376d30dbee447)  (db0)

## [v2.7.6](https://github.com/Haidra-Org/hordelib/compare/v2.7.5...v2.7.6)

23 March 2024

- fix: give up on ti hash mismatches [`2da5b9b`](https://github.com/Haidra-Org/hordelib/commit/2da5b9b4415d7c343aba2e588c1e8406ec9af26e)  (tazlin)
- fix: remove safety_checker from default [`ed9d1c7`](https://github.com/Haidra-Org/hordelib/commit/ed9d1c74cdb08281554f6a3049a824af3183deb2)  (tazlin)
- feat: support comfyui `a28a9dc` [`2060e0e`](https://github.com/Haidra-Org/hordelib/commit/2060e0e6b685de56b666773a63f8441c1e4f1d0b)  (tazlin)

## [v2.7.5](https://github.com/Haidra-Org/hordelib/compare/v2.7.4...v2.7.5)

21 March 2024

- fix: don't crash on missing source image [`706e7d8`](https://github.com/Haidra-Org/hordelib/commit/706e7d85d495d627978541bea03c7d5dd4bad705)  (tazlin)

## [v2.7.4](https://github.com/Haidra-Org/hordelib/compare/v2.7.3...v2.7.4)

21 March 2024

- fix: callback during each step of post-proc. [`f0e2fc1`](https://github.com/Haidra-Org/hordelib/commit/f0e2fc131dd919e52fba06b865253b966b399005)  (tazlin)

## [v2.7.3](https://github.com/Haidra-Org/hordelib/compare/v2.7.2...v2.7.3)

20 March 2024

- Revert "fix: use known working version of comfyui" [`08243f8`](https://github.com/Haidra-Org/hordelib/commit/08243f8bd437d6d9eeccb35d1e482d2326d5d207)  (tazlin)

## [v2.7.2](https://github.com/Haidra-Org/hordelib/compare/v2.7.1...v2.7.2)

20 March 2024

- fix: use known working version of comfyui [`fde1a6d`](https://github.com/Haidra-Org/hordelib/commit/fde1a6d74e68bd39cf581687751941d894195e6c)  (tazlin)

## [v2.7.1](https://github.com/Haidra-Org/hordelib/compare/v2.7.0...v2.7.1)

20 March 2024

- feat: Support Stable Cascade Img2Img [`1e1406d`](https://github.com/Haidra-Org/hordelib/commit/1e1406d5e8be94f83983422787531c4e2b00345a)  (db0)
- tests: adds remix test image checks + comparison images [`80a3489`](https://github.com/Haidra-Org/hordelib/commit/80a348930b885579bb71dfdaf8c49470f37acbcd)  (tazlin)
- feat: allow adhoc lora downloads to run concurrently [`1c5a54f`](https://github.com/Haidra-Org/hordelib/commit/1c5a54f3d9fa32b42ca4ec4dedcda7614718f0bb)  (tazlin)

## [v2.7.0](https://github.com/Haidra-Org/hordelib/compare/v2.6.5...v2.7.0)

12 March 2024

- feat: support callbacks for inference/post-proc. [`d13e782`](https://github.com/Haidra-Org/hordelib/commit/d13e78254a7b8a32a76340a8bcfd54658800e23b)  (tazlin)
- fix: try disabling grad on chkpoint load [`f3d5cc0`](https://github.com/Haidra-Org/hordelib/commit/f3d5cc001f9835ffb6d93f9f2547731783d708e8)  (tazlin)
- feat: comfyui to `2a813c3b` [`ec32ca1`](https://github.com/Haidra-Org/hordelib/commit/ec32ca13cdb999abce6ab011c791eaef30bfb9be)  (tazlin)

## [v2.6.5](https://github.com/Haidra-Org/hordelib/compare/v2.6.4...v2.6.5)

9 March 2024

- fix: check KNOWN_UPSCALER membership by value too [`4488ac2`](https://github.com/Haidra-Org/hordelib/commit/4488ac2e6a5285c8ac690633fca792cd9d105fbb)  (tazlin)

## [v2.6.4](https://github.com/Haidra-Org/hordelib/compare/v2.6.3...v2.6.4)

7 March 2024

- fix: increase timeout rate of lora metadata/model downloads [`9a1b630`](https://github.com/Haidra-Org/hordelib/commit/9a1b6308027d37c77ece303157e94a980bd57e8f)  (tazlin)
- fix: retry 500s a few times on lora/ti metadata dl timeout [`c07d315`](https://github.com/Haidra-Org/hordelib/commit/c07d315619b14fa4c997f08c808322da79398554)  (tazlin)
- fix: retry less often with TI model manager also [`31deb1e`](https://github.com/Haidra-Org/hordelib/commit/31deb1ec22b5bf15be2bf95d5808913951640780)  (tazlin)

## [v2.6.3](https://github.com/Haidra-Org/hordelib/compare/v2.6.2...v2.6.3)

5 March 2024

- fix: handle 401s for lora dls as terminal; spend less time retrying [`59cb36c`](https://github.com/Haidra-Org/hordelib/commit/59cb36c384f6a4ffc9a4e94db9ebc04f642001a8)  (tazlin)

## [v2.6.2](https://github.com/Haidra-Org/hordelib/compare/v2.6.1...v2.6.2)

24 February 2024

- feat: Passes CivitAI token on checkpoint download as well [`#202`](https://github.com/Haidra-Org/hordelib/pull/202) (Divided by Zer0)

## [v2.6.1](https://github.com/Haidra-Org/hordelib/compare/v2.6.0...v2.6.1)

24 February 2024

- fix: handle missing inpainting source images/masks better [`7a690db`](https://github.com/Haidra-Org/hordelib/commit/7a690db4377d5a27fa35556f24854a5acb3962ba)  (tazlin)
- refactor: clarify (in code and in logs) payload "model" parsing [`a9a34d6`](https://github.com/Haidra-Org/hordelib/commit/a9a34d6d97079007be4948bdddfa3d4daea916ec)  (tazlin)
- feat: free resources less often [`1bb9279`](https://github.com/Haidra-Org/hordelib/commit/1bb92792f4b4981e7a328922bc73768c1028c5e0)  (tazlin)

## [v2.6.0](https://github.com/Haidra-Org/hordelib/compare/v2.5.3...v2.6.0)

22 February 2024

- Feat: Adds support for Stable Cascade [`#195`](https://github.com/Haidra-Org/hordelib/pull/195) (Divided by Zer0)

## [v2.5.3](https://github.com/Haidra-Org/hordelib/compare/v2.5.2...v2.5.3)

19 February 2024

## [v2.5.2](https://github.com/Haidra-Org/hordelib/compare/v2.5.1...v2.5.2)

19 February 2024

- fix: attempt to reload the json on fail [`bef5013`](https://github.com/Haidra-Org/hordelib/commit/bef5013b27c0f57ed9d1cd5f819e4c91b63026bc)  (db0)
- tests: skip sde check during CI [`85ccde3`](https://github.com/Haidra-Org/hordelib/commit/85ccde35d46b86805431afed0b662591b93c7cc2)  (tazlin)
- tests: reorder test execution order [`6f201ee`](https://github.com/Haidra-Org/hordelib/commit/6f201eea155e114d33d5d382c80c29839cb9c2f9)  (tazlin)

## [v2.5.1](https://github.com/Haidra-Org/hordelib/compare/v2.5.0...v2.5.1)

6 February 2024

- fix: remove `torchaudio` from req. deps [`86c239e`](https://github.com/Haidra-Org/hordelib/commit/86c239e6a3a4d5b280537e51167c0befa9619d3e)  (tazlin)

## [v2.5.0](https://github.com/Haidra-Org/hordelib/compare/v2.4.2...v2.5.0)

6 February 2024

- feat: initial pyinstaller support [`05c6548`](https://github.com/Haidra-Org/hordelib/commit/05c65487897c49f2b86f1b393f215b962944c94e)  (tazlin)
- chore: update pre-commit hook versions [`1fd63a5`](https://github.com/Haidra-Org/hordelib/commit/1fd63a55fcdfd366d21920e3fdd4e5b434e2fe64)  (tazlin)
- build(deps-dev): bump the python-packages group with 4 updates [`c161575`](https://github.com/Haidra-Org/hordelib/commit/c161575d8be5a2517414db7b6da1238fb3509e47)  (dependabot[bot])

## [v2.4.2](https://github.com/Haidra-Org/hordelib/compare/v2.4.1...v2.4.2)

24 January 2024

- fix: Attempt to stop loras going randomly missing [`18bea86`](https://github.com/Haidra-Org/hordelib/commit/18bea86652b2ff186400e6347b3d3940c7e04c34)  (db0)
- fix: Avoid deadlock with mutex [`ac8fe8c`](https://github.com/Haidra-Org/hordelib/commit/ac8fe8c596093de0306ae57592730a171bda24bf)  (db0)

## [v2.4.1](https://github.com/Haidra-Org/hordelib/compare/v2.4.0...v2.4.1)

24 January 2024

- tests: add download gated lora test [`2541494`](https://github.com/Haidra-Org/hordelib/commit/25414940740b228821258ecc36a6caf857ece345)  (tazlin)
- feat: Add CivitAI token while downloading Loras and ti's [`5a7f221`](https://github.com/Haidra-Org/hordelib/commit/5a7f221aa9e8d21b52665b4934a04f9a2545d075)  (Gabriel Janczak)
- fix: removing unnessesery constructor parameters and basic test [`3af9813`](https://github.com/Haidra-Org/hordelib/commit/3af9813dab125fca926f750c7c04bc3a66eac76e)  (Gabriel Janczak)

## [v2.4.0](https://github.com/Haidra-Org/hordelib/compare/v2.3.7...v2.4.0)

11 January 2024

- feat: Allow img2img to use n_iter [`#158`](https://github.com/Haidra-Org/hordelib/pull/158) (Divided by Zer0)
- build(deps): bump the python-packages group with 7 updates [`12f2293`](https://github.com/Haidra-Org/hordelib/commit/12f22938f9ea6d25df8c9719f1607481ecf4fc74)  (dependabot[bot])
- chore: use latest pre-commit hook versions [`2f2adfe`](https://github.com/Haidra-Org/hordelib/commit/2f2adfe33defc9756a7808ce090391b908bb8036)  (tazlin)

## [v2.3.7](https://github.com/Haidra-Org/hordelib/compare/v2.3.6...v2.3.7)

9 January 2024

- fix: try and fallback to on-disk model ref when can't download [`df98872`](https://github.com/Haidra-Org/hordelib/commit/df9887225e1b09b86b846e5fd530551d145117bb)  (tazlin)
- docs: add docstring for `load_model_managers` [`5662e10`](https://github.com/Haidra-Org/hordelib/commit/5662e1092eb22d27fec3075232024be1b1a0a065)  (tazlin)
- fix: demote log message level for no-download-required refs [`cedd974`](https://github.com/Haidra-Org/hordelib/commit/cedd9744127aeabf846721799d7d00764bc4b19c)  (tazlin)

## [v2.3.6](https://github.com/Haidra-Org/hordelib/compare/v2.3.5...v2.3.6)

4 January 2024

- fix: orig name not searched in lowercase [`688de0f`](https://github.com/Haidra-Org/hordelib/commit/688de0f383f6d0f12a3a0356eca8799d13a79410)  (db0)

## [v2.3.5](https://github.com/Haidra-Org/hordelib/compare/v2.3.4...v2.3.5)

2 January 2024

- feat: ensure we check for lora refresh regularly [`9df10d7`](https://github.com/Haidra-Org/hordelib/commit/9df10d71a391245f3d6a9ef2ec983ae29a5f7c9e)  (db0)
- fix: requesting generic lora not returning the latest version [`6bb4789`](https://github.com/Haidra-Org/hordelib/commit/6bb4789324b1117363c9ba14cb1cbbee1b4e6a4f)  (db0)
- fix: more logging in lora exceptions [`4d03128`](https://github.com/Haidra-Org/hordelib/commit/4d031286d08e4a20d6cfa9660399ef7ca7277ceb)  (tazlin)

## [v2.3.4](https://github.com/Haidra-Org/hordelib/compare/v2.3.3...v2.3.4)

30 December 2023

- ci: test comfyui pipeline failures [`85a065f`](https://github.com/Haidra-Org/hordelib/commit/85a065f0569b48ba84702057b4461847349c7194)  (tazlin)
- ci: test `n_iter` [`802e6e7`](https://github.com/Haidra-Org/hordelib/commit/802e6e7fe07b0deb2ce8446ddec57d7642672be6)  (tazlin)
- fix: be more stringent checking comfyui's output [`566592a`](https://github.com/Haidra-Org/hordelib/commit/566592a1863d41ec0aa3ba0cf53ffc7f10a63766)  (tazlin)

## [v2.3.3](https://github.com/Haidra-Org/hordelib/compare/v2.3.2...v2.3.3)

29 December 2023

- ci: run lora tests sooner; samplers test later [`#139`](https://github.com/Haidra-Org/hordelib/pull/139) (tazlin)
- fix: downloading loras on fast systems no longer causes duplicate images [`#133`](https://github.com/Haidra-Org/hordelib/pull/133) (Divided by Zer0)
- ci: don't hang on lora setup_and_teardown [`17d8274`](https://github.com/Haidra-Org/hordelib/commit/17d82742f5f3793793a7819e1c21928b2034d39f)  (tazlin)
- fix: check for more invalid lora names [`0ddd4e9`](https://github.com/Haidra-Org/hordelib/commit/0ddd4e9d846c81dd5088a7ba40da09991ca4bb50)  (tazlin)

## [v2.3.2](https://github.com/Haidra-Org/hordelib/compare/v2.3.1...v2.3.2)

28 December 2023

- fix: purge caches for comfyui executor [`048e856`](https://github.com/Haidra-Org/hordelib/commit/048e856f5da12e4d98d2a46900b1d862a44090f2)  (tazlin)

## [v2.3.1](https://github.com/Haidra-Org/hordelib/compare/v2.3.0...v2.3.1)

28 December 2023

- fix: support comfyui args; detect pipeline changes between runs [`5089ac2`](https://github.com/Haidra-Org/hordelib/commit/5089ac2e7fbf465d93e5ddeb35b5d79f307b011f)  (tazlin)
- ci: don't test `k_dpmpp_sde` so stringently [`1d9c938`](https://github.com/Haidra-Org/hordelib/commit/1d9c93812bbef8b873ceb80b5d7c49f0540a4448)  (tazlin)
- feat: update to comfy version `c78214` [`8dff541`](https://github.com/Haidra-Org/hordelib/commit/8dff5413d190ef5dae6e05272d12e6cd1f859093)  (tazlin)

## [v2.3.0](https://github.com/Haidra-Org/hordelib/compare/v2.2.4...v2.3.0)

27 December 2023

- feat: allows multiple lora versions [`#127`](https://github.com/Haidra-Org/hordelib/pull/127) (Divided by Zer0)

## [v2.2.4](https://github.com/Haidra-Org/hordelib/compare/v2.2.3...v2.2.4)

13 December 2023

- fix: retry hordeling more often; try fewer times on 500s [`ff33292`](https://github.com/Haidra-Org/hordelib/commit/ff3329289f545f74a7bb10eb20b20fb183a9bab5)  (tazlin)
- fix: do download timeout checks for TIs too [`3155c99`](https://github.com/Haidra-Org/hordelib/commit/3155c994fcc3c95d609ca0ab2fe40e5b39cd181d)  (tazlin)

## [v2.2.3](https://github.com/Haidra-Org/hordelib/compare/v2.2.2...v2.2.3)

9 December 2023

- fix: must initiate final_rawpng [`2a3483d`](https://github.com/Haidra-Org/hordelib/commit/2a3483d2b0dce6edc6544d0072a6ce3b1eb550db)  (db0)

## [v2.2.2](https://github.com/Haidra-Org/hordelib/compare/v2.2.1...v2.2.2)

9 December 2023

- fix: Handle missing lora IDs [`2948c58`](https://github.com/Haidra-Org/hordelib/commit/2948c5899d783a69fd77ecb502b5c7b6903a4dbd)  (db0)

## [v2.2.1](https://github.com/Haidra-Org/hordelib/compare/v2.2.0...v2.2.1)

6 December 2023

- feat: update to comfy version `e13454` (SDV support) [`99dcdb1`](https://github.com/Haidra-Org/hordelib/commit/99dcdb12234a82e16bd59a78391729223de63a1f)  (tazlin)

## [v2.2.0](https://github.com/Haidra-Org/hordelib/compare/v2.1.1...v2.2.0)

5 December 2023

- feat: Inference and PP return objects instead of Images [`#96`](https://github.com/Haidra-Org/hordelib/pull/96) (Divided by Zer0)
- build(deps): bump the python-packages group with 6 updates [`4cbc080`](https://github.com/Haidra-Org/hordelib/commit/4cbc080310560ae7f2bb8e7d665fe1d98fec9643)  (dependabot[bot])
- chore: add dependabot support [`6b6c001`](https://github.com/Haidra-Org/hordelib/commit/6b6c0018b36a0a3008f4ff6a78acb6bc4ef8ed37)  (tazlin)
- chore: update pre-commit hooks [`f2808cc`](https://github.com/Haidra-Org/hordelib/commit/f2808cca309cd44b3118871002087dd7caede794)  (tazlin)

## [v2.1.1](https://github.com/Haidra-Org/hordelib/compare/v2.1.0...v2.1.1)

10 November 2023

- tests: fix: adds checks for missing lora image comparisons [`55af765`](https://github.com/Haidra-Org/hordelib/commit/55af765dc777ebb99e6ade86fab908a317221c45)  (tazlin)
- fix: write out a copy of `model_reference` in lora MM [`6d46a75`](https://github.com/Haidra-Org/hordelib/commit/6d46a75d1ec23123d382ca8a27eae3ef21965a9c)  (tazlin)
- tests: fix: print out failing image for samplers check [`9f3c92c`](https://github.com/Haidra-Org/hordelib/commit/9f3c92c900b18bd2a1b6509285fa6eba2cb613d8)  (tazlin)

## [v2.1.0](https://github.com/Haidra-Org/hordelib/compare/v2.0.2...v2.1.0)

2 November 2023

- feat: graceful fail if we detect a login redirect [`930a6f9`](https://github.com/Haidra-Org/hordelib/commit/930a6f95c047d8ff3c38f12f0475d198e934d271)  (db0)
- feat: use comfy dd116abfc48e8023bb425c2dd5bd954ee99d7a9c [`63954b4`](https://github.com/Haidra-Org/hordelib/commit/63954b4de9f802a20f86931dd13609217dfb0de2)  (tazlin)
- docs: fix reference to old cuda version [`97b83e3`](https://github.com/Haidra-Org/hordelib/commit/97b83e30188f0018365ebf3c26df074f6dde964e)  (tazlin)

## [v2.0.2](https://github.com/Haidra-Org/hordelib/compare/v2.0.1...v2.0.2)

5 October 2023

- fix: more relaxed memory management (allows high vram?) [`ed3120a`](https://github.com/Haidra-Org/hordelib/commit/ed3120a23d2986564bf9aff9605a0f9576d71e16)  (tazlin)
- chore: latest comfyui version [`d8961bb`](https://github.com/Haidra-Org/hordelib/commit/d8961bb2bd20266e970c31ced8a008c218a05817)  (tazlin)
- feat: use torch 2.1 + CU121 [`e6955ea`](https://github.com/Haidra-Org/hordelib/commit/e6955eaf43ead5d02e4d074e74c3c686b8a34353)  (tazlin)

## [v2.0.1](https://github.com/Haidra-Org/hordelib/compare/v2.0.0...v2.0.1)

4 October 2023

- feat: add `AIWORKER_LORA_CACHE_SIZE` env var [`582747d`](https://github.com/Haidra-Org/hordelib/commit/582747d1ba328933ed4fc4b65ba5607e5d92b906)  (tazlin)
- fix: handle lora env var `None` [`7f4c6f5`](https://github.com/Haidra-Org/hordelib/commit/7f4c6f5679ee35ad00b570460a26e5fd30f0a543)  (tazlin)

# [v2.0.0](https://github.com/Haidra-Org/hordelib/compare/v1.6.6...v2.0.0)

3 October 2023

- fix: remove any load to ram/vram code, make hordelib single threaded [`98178f3`](https://github.com/Haidra-Org/hordelib/commit/98178f3b983b2887511431372f8194a1a5b3affe)  (tazlin)
- refactor: comfy handles memory; better multiprocessing scaffolding [`775ecf6`](https://github.com/Haidra-Org/hordelib/commit/775ecf60b9b89966bba8e4fe969a0ddb6d6a1ae4)  (tazlin)
- feat: worker beta changes [`f6bc72e`](https://github.com/Haidra-Org/hordelib/commit/f6bc72e3a95f935593cd54ac273f5a9c9920db91)  (tazlin)

## [v1.6.6](https://github.com/Haidra-Org/hordelib/compare/v1.6.5...v1.6.6)

6 September 2023

- fix: inject negative embeddings correctly [`3608dd5`](https://github.com/Haidra-Org/hordelib/commit/3608dd57681368cb47cf81ef1d0a52257fddfdbb)  (tazlin)
- fix: (ti inject) don't strip pre-existing trailing comma from neg prompt [`850bdd5`](https://github.com/Haidra-Org/hordelib/commit/850bdd50f277b27d67b56746846eba54316b9acb)  (tazlin)
- fix: missing image resize [`0a08c5e`](https://github.com/Haidra-Org/hordelib/commit/0a08c5e7d9767e4fadeb9cfec589f6f643117a2b)  (tazlin)

## [v1.6.5](https://github.com/Haidra-Org/hordelib/compare/v1.6.4...v1.6.5)

27 August 2023

- feat: Automatic downloading of TIs [`#55`](https://github.com/Haidra-Org/hordelib/pull/55) (Divided by Zer0)
- fix: TI now correctly compares to intended sha256 from hordeling [`45ca52d`](https://github.com/Haidra-Org/hordelib/commit/45ca52d035ff15e6b5a3e8280d645600bce513a5)  (tazlin)

## [v1.6.4](https://github.com/Haidra-Org/hordelib/compare/v1.6.3...v1.6.4)

2 August 2023

- fix: release yaml oversight [`6138e94`](https://github.com/Haidra-Org/hordelib/commit/6138e94501dae3779cf7569a594edb60d169ba1c)  (tazlin)

## [v1.6.3](https://github.com/Haidra-Org/hordelib/compare/v1.6.2...v1.6.3)

2 August 2023

- feat: Loras random "any" and allows "all" trigger [`#56`](https://github.com/Haidra-Org/hordelib/pull/56) (Efreak)
- refactor: mypy assisted cleanup, py 3.11 support [`9912d03`](https://github.com/Haidra-Org/hordelib/commit/9912d033ef826f55214c33f782b78e7084540998)  (tazlin)
- tests: warm loaded models now are unloaded at test run start [`749e577`](https://github.com/Haidra-Org/hordelib/commit/749e5770b9c2ae449f8f6552ee75f5eeb34a3c9a)  (tazlin)
- Tests: negative lora model strength values [`f91ae22`](https://github.com/Haidra-Org/hordelib/commit/f91ae225fc34ffa8bf869c06ac717b79064d8267)  (db0)

## [v1.6.2](https://github.com/Haidra-Org/hordelib/compare/v1.6.1...v1.6.2)

25 June 2023

- fix: quiet log spam from `get_mm_pointers(...)` [`4301d0c`](https://github.com/Haidra-Org/hordelib/commit/4301d0cc1b4f5ccd251ee26230dabf76cac24a28)  (tazlin)

## [v1.6.1](https://github.com/Haidra-Org/hordelib/compare/v1.6.0...v1.6.1)

22 June 2023

- fix: re-add taint models (unintentionally removed in 1.6.0) [`2b7dbbd`](https://github.com/Haidra-Org/hordelib/commit/2b7dbbde472ed52a82404b52854bbdcc5b64d290)  (tazlin)
- fix: correctly require correct versions of horde_* deps [`c6fb00b`](https://github.com/Haidra-Org/hordelib/commit/c6fb00b307f863a9303b2e8c54ba8433c0e52841)  (tazlin)

## [v1.6.0](https://github.com/Haidra-Org/hordelib/compare/v1.5.2...v1.6.0)

21 June 2023

- refactor: prefer load MMs by enum/type, deprecate named MM params [`4505d89`](https://github.com/Haidra-Org/hordelib/commit/4505d8948f3522b510b6548187127166f09747c0)  (tazlin)
- refactor: rework SharedModelManager tests to `conftest.py` driven fixtures [`8694513`](https://github.com/Haidra-Org/hordelib/commit/8694513f5370bbd2f8f6ee17be24e8cf664c0b19)  (tazlin)
- refactor: rework post processor tests to `conftest.py` driven fixtures [`182cb3c`](https://github.com/Haidra-Org/hordelib/commit/182cb3c6722baf030d348b4a75dbe7c5d107bed5)  (tazlin)

## [v1.5.2](https://github.com/Haidra-Org/hordelib/compare/v1.5.1...v1.5.2)

12 June 2023

- fix: don't cache incorrect location of model directory [`#14`](https://github.com/Haidra-Org/hordelib/pull/14) (Jug)

## [v1.5.1](https://github.com/Haidra-Org/hordelib/compare/v1.5.0...v1.5.1)

10 June 2023

- chore: use clipfree as a pypi package ('horde_clipfree') [`#11`](https://github.com/Haidra-Org/hordelib/pull/11) (tazlin)

## [v1.5.0](https://github.com/Haidra-Org/hordelib/compare/v1.4.0...v1.5.0)

10 June 2023

- ci: re-enable pypi release publishing [`#9`](https://github.com/Haidra-Org/hordelib/pull/9) (Jug)
- docs: update readme [`#8`](https://github.com/Haidra-Org/hordelib/pull/8) (Jug)
- fix: make finding the model directory backwards compatible [`#7`](https://github.com/Haidra-Org/hordelib/pull/7) (Jug)
- fix: clipfree compatability fixes [`#5`](https://github.com/Haidra-Org/hordelib/pull/5) (tazlin)
- feat: Re-aded clip/blip based on external clipfree library [`#2`](https://github.com/Haidra-Org/hordelib/pull/2) (Divided by Zer0)
- ci: fix PR tests image deployment [`#3`](https://github.com/Haidra-Org/hordelib/pull/3) (Jug)

## [v1.4.0](https://github.com/Haidra-Org/hordelib/compare/v1.3.17...v1.4.0)

4 June 2023

- feat: disable pypi publishing but enable auto tests [`b4f5bf9`](https://github.com/Haidra-Org/hordelib/commit/b4f5bf98c6a43e85852078c86071340ef54f0cde)  (Jug)
- doc: update other readme urls [`e38ac87`](https://github.com/Haidra-Org/hordelib/commit/e38ac879735a0b2e0727bde8171be314b56b6204)  (Jug)
- doc: add note about version divergence [`f559c5d`](https://github.com/Haidra-Org/hordelib/commit/f559c5d4bf52e972467ad702dd1b1f7f56544d5f)  (Jug)

## [v1.3.17](https://github.com/Haidra-Org/hordelib/compare/v1.3.10...v1.3.17)

4 June 2023

- fix: tighten up thread safety around the sampler [`#341`](https://github.com/Haidra-Org/hordelib/pull/341) (Jug)
- fix: don't return results of another job in certain corner cases. [`#339`](https://github.com/Haidra-Org/hordelib/pull/339) (Jug)
- fix: remove any reference to blip, clip or cache. [`76011bd`](https://github.com/Haidra-Org/hordelib/commit/76011bda396baa5cda1281b252a0db3bce55a007)  (Jug)
- feat: resync with hordelib 1.3.17 [`357ec15`](https://github.com/Haidra-Org/hordelib/commit/357ec155d778d56d888b9ab9be4ebfaeeff44f05)  (Jug)

## [v1.3.10](https://github.com/Haidra-Org/hordelib/compare/v1.3.9...v1.3.10)

29 May 2023

- fix: handles lora name being sent as a string int [`3b74795`](https://github.com/Haidra-Org/hordelib/commit/3b74795cbe85c904dc4ee0e6a83287ca658a90b3)  (db0)

## [v1.3.9](https://github.com/Haidra-Org/hordelib/compare/v1.3.8...v1.3.9)

29 May 2023

- feat: Add seeking loras by ID and unicode [`228c8cd`](https://github.com/Haidra-Org/hordelib/commit/228c8cd3070c375743f79d8465411dc82e14c392)  (db0)
- fix: avoid crash when resetting adhoc loras [`d4a21e5`](https://github.com/Haidra-Org/hordelib/commit/d4a21e593e29eb46833cec5e7335dc74171ba534)  (db0)

## [v1.3.8](https://github.com/Haidra-Org/hordelib/compare/v1.3.7...v1.3.8)

27 May 2023

- fix: logging error with loading cnet [`#332`](https://github.com/Haidra-Org/hordelib/pull/332) (Jug)

## [v1.3.7](https://github.com/Haidra-Org/hordelib/compare/v1.3.6...v1.3.7)

27 May 2023

- feat: keeping some unused loras as adhoc [`f66db4d`](https://github.com/Haidra-Org/hordelib/commit/f66db4dff4a8c01e548a33c2963612d48bfb6a20)  (db0)
- feat: More robust tracking of lora downloads [`4473c5b`](https://github.com/Haidra-Org/hordelib/commit/4473c5b5848a28a5429de64667b434a9749e722f)  (db0)
- tests: integration with TESTS_ONGOING os env [`101761e`](https://github.com/Haidra-Org/hordelib/commit/101761e191a6fc65761eee9900c9a6f6a00a6757)  (db0)

## [v1.3.6](https://github.com/Haidra-Org/hordelib/compare/v1.3.5...v1.3.6)

26 May 2023

- tests: Added test for lora model_reference wipe [`a63f896`](https://github.com/Haidra-Org/hordelib/commit/a63f896a74eddfaffa1acb7d5a5d4f538ce429ee)  (db0)
- fix: wipe reference only when valid [`2d6a7bc`](https://github.com/Haidra-Org/hordelib/commit/2d6a7bc0debe7976d1c6e2795fb8e3536d7ee5a4)  (db0)
- feat: add changelog link to release annoucement [`8357cb0`](https://github.com/Haidra-Org/hordelib/commit/8357cb0fe1438be4bf7289f392551b219e2aa7dd)  (Jug)

## [v1.3.5](https://github.com/Haidra-Org/hordelib/compare/v1.3.4...v1.3.5)

25 May 2023

- fix: allow all types of downloads to display progress [`#324`](https://github.com/Haidra-Org/hordelib/pull/324) (Jug)
- fix: make index uses .png files [`#321`](https://github.com/Haidra-Org/hordelib/pull/321) (Divided by Zer0)

## [v1.3.4](https://github.com/Haidra-Org/hordelib/compare/v1.3.3...v1.3.4)

25 May 2023

- tests: Compare test images with expected output [`#319`](https://github.com/Haidra-Org/hordelib/pull/319) (Divided by Zer0)
- feat: add support for download progress indicators [`#318`](https://github.com/Haidra-Org/hordelib/pull/318) (Jug)
- ci: set IMAGE_DISTANCE_THRESHOLD [`48a5a6e`](https://github.com/Haidra-Org/hordelib/commit/48a5a6ea455785e1b8ddb567f7acae5258bdd0d0)  (db0)
- doc: restore PR unit test image link [`40aa2bd`](https://github.com/Haidra-Org/hordelib/commit/40aa2bd10978c2e48985d61c6e81ec0faea54a71)  (Jug)
- doc: remove link to PR image tests which were removed [`f79b737`](https://github.com/Haidra-Org/hordelib/commit/f79b737fdb60a765518561d96bd756c7f248e979)  (Jug)

## [v1.3.3](https://github.com/Haidra-Org/hordelib/compare/v1.3.2...v1.3.3)

25 May 2023

- fix: make fakescribble controlnet work again [`#314`](https://github.com/Haidra-Org/hordelib/pull/314) (Jug)

## [v1.3.2](https://github.com/Haidra-Org/hordelib/compare/v1.3.1...v1.3.2)

25 May 2023

- fix: hangs and random processing results with multiple threads regression [`#311`](https://github.com/Haidra-Org/hordelib/pull/311) (Jug)
- fix: ensure lora folder exists before starting download [`#309`](https://github.com/Haidra-Org/hordelib/pull/309) (Divided by Zer0)

## [v1.3.1](https://github.com/Haidra-Org/hordelib/compare/v1.3.0...v1.3.1)

24 May 2023

- fix: more robust downloads; resume, retry, don't delete files so hastily.  [`#307`](https://github.com/Haidra-Org/hordelib/pull/307) (Jug)

## [v1.3.0](https://github.com/Haidra-Org/hordelib/compare/v1.2.1...v1.3.0)

24 May 2023

- fix: moved lora downloads outside of init [`#304`](https://github.com/Haidra-Org/hordelib/pull/304) (Divided by Zer0)
- Lora Model Manager [`#302`](https://github.com/Haidra-Org/hordelib/pull/302) (Divided by Zer0)
- fix: fix some tests and update docs for Linux [`#301`](https://github.com/Haidra-Org/hordelib/pull/301) (Jug)
- feat: Added trigger injection to loras [`7ef0745`](https://github.com/Haidra-Org/hordelib/commit/7ef0745654d6b802fefbf796a6198e402012ff9b)  (db0)
- feat: allow searching triggers [`acf753b`](https://github.com/Haidra-Org/hordelib/commit/acf753b96bbe3413ea743ba79cce170302f5fd7e)  (db0)
- fix: tweak lora tests and node loader [`f7db62c`](https://github.com/Haidra-Org/hordelib/commit/f7db62cac609164575d98c64cea121610abf3e4e)  (Jug)

## [v1.2.1](https://github.com/Haidra-Org/hordelib/compare/v1.2.0...v1.2.1)

22 May 2023

- fix: remove "No job ran for x seconds" warning [`#298`](https://github.com/Haidra-Org/hordelib/pull/298) (Jug)
- fix: ignore unknown loras, search case insensitively for them [`#297`](https://github.com/Haidra-Org/hordelib/pull/297) (Jug)

## [v1.2.0](https://github.com/Haidra-Org/hordelib/compare/v1.1.2...v1.2.0)

21 May 2023

- fix: unit tests use about 6GB VRAM max now. [`#293`](https://github.com/Haidra-Org/hordelib/pull/293) (Jug)
- feat: refactor for clarity, tweak img2img and inpainting, tidy tests [`#290`](https://github.com/Haidra-Org/hordelib/pull/290) (Jug)
- Add alt pipeline design for img2img with mask [`#279`](https://github.com/Haidra-Org/hordelib/pull/279) (Wolfgang Meyers)

## [v1.1.2](https://github.com/Haidra-Org/hordelib/compare/v1.1.1...v1.1.2)

19 May 2023

## [v1.1.1](https://github.com/Haidra-Org/hordelib/compare/v1.1.0...v1.1.1)

19 May 2023

- fix: minimum version of horde_model_reference [`a603d0b`](https://github.com/Haidra-Org/hordelib/commit/a603d0be7c18e98e122293ba36b5ea034e2375d7)  (tazlin)
- fix: typo in minimum requirement [`d997156`](https://github.com/Haidra-Org/hordelib/commit/d99715647562ac8ef9872d2c49eec347d7a12b82)  (tazlin)

## [v1.1.0](https://github.com/Haidra-Org/hordelib/compare/v1.0.5...v1.1.0)

19 May 2023

- fix: correctly output pipeline json during development [`#284`](https://github.com/Haidra-Org/hordelib/pull/284) (Jug)
- fix: auto fix bad cfg values [`#282`](https://github.com/Haidra-Org/hordelib/pull/282) (Jug)
- feat: add lora support and reduce cnet memory requirements by 50% [`#270`](https://github.com/Haidra-Org/hordelib/pull/270) (Jug)

## [v1.0.5](https://github.com/Haidra-Org/hordelib/compare/v1.0.4...v1.0.5)

19 May 2023

- refactor: get_mm_pointers accommodates `type` as well [`b97e7d3`](https://github.com/Haidra-Org/hordelib/commit/b97e7d3ceae1ac265541d5c1fc825e0bde2c71f3)  (tazlin)
- fix: unsupport 'diffusers' [`0d84848`](https://github.com/Haidra-Org/hordelib/commit/0d8484832f2e1a7a803609d343d940f01d9f1307)  (tazlin)
- feat: move stable_diffusion_inpainting if in diffusers directory [`3193420`](https://github.com/Haidra-Org/hordelib/commit/31934201482cf4e548569cbcd16bed3267cf5f05)  (tazlin)

## [v1.0.4](https://github.com/Haidra-Org/hordelib/compare/v1.0.3...v1.0.4)

18 May 2023

- Increase read/write sizes during download/checksums [`#274`](https://github.com/Haidra-Org/hordelib/pull/274) (Andy Pilate)
- feat: support prepending proxy URL to github downloads [`9783787`](https://github.com/Haidra-Org/hordelib/commit/9783787777a9a7123c27ccbe2849371a5814bf8e)  (tazlin)
- ci: fix: allow release workflow repo write permissions [`3ed0733`](https://github.com/Haidra-Org/hordelib/commit/3ed0733fb6f73caf1888f83fad18ff7c866cd6d3)  (tazlin)

## [v1.0.3](https://github.com/Haidra-Org/hordelib/compare/v1.0.2...v1.0.3)

15 May 2023

- When gathering loaded/available names, allows filtering by model manager type [`#254`](https://github.com/Haidra-Org/hordelib/pull/254) (Divided by Zer0)
- feat: upgrade to the latest comfyui [`#255`](https://github.com/Haidra-Org/hordelib/pull/255) (Jug)
- feat: add option to enable/disable batch optimisation [`#252`](https://github.com/Haidra-Org/hordelib/pull/252) (Jug)
- fix: untrack automatically downloaded model reference jsons [`501f35d`](https://github.com/Haidra-Org/hordelib/commit/501f35d897153bb95ef50c32be8a3045449fb4c1)  (tazlin)
- fix: remove unused model 'db.json' [`6a5b29d`](https://github.com/Haidra-Org/hordelib/commit/6a5b29dc005e30f854b09070348c30fcbb7a5638)  (tazlin)
- fix: ignore automatically downloaded model references [`ff0428d`](https://github.com/Haidra-Org/hordelib/commit/ff0428dfecc6c0809b4e652cf44797575ccb6399)  (tazlin)

## [v1.0.2](https://github.com/Haidra-Org/hordelib/compare/v1.0.1...v1.0.2)

14 May 2023

- fix: correctly unload models from gpu under stress [`#249`](https://github.com/Haidra-Org/hordelib/pull/249) (Jug)

## [v1.0.1](https://github.com/Haidra-Org/hordelib/compare/v1.0.0...v1.0.1)

14 May 2023

- fix: benchmark looks harder for model directory [`#247`](https://github.com/Haidra-Org/hordelib/pull/247) (Jug)
- doc: remove changelog from main [`682ec5e`](https://github.com/Haidra-Org/hordelib/commit/682ec5e63dcf779ed6496fe4762fcc39ffdfc203)  (Jug)
- doc: update readme [`6834c61`](https://github.com/Haidra-Org/hordelib/commit/6834c61549cb442d0a9ab00929c3991e7e05ca98)  (Jug)
- doc: update readme [`101a8b5`](https://github.com/Haidra-Org/hordelib/commit/101a8b5867a3ea15f2f7cb683b00d2af33b1d736)  (Jug)

# [v1.0.0](https://github.com/Haidra-Org/hordelib/compare/v0.19.15...v1.0.0)

14 May 2023

- feat: release v1.0.0 [`#246`](https://github.com/Haidra-Org/hordelib/pull/246) (Jug)

## [v0.19.15](https://github.com/Haidra-Org/hordelib/compare/v0.19.14...v0.19.15)

14 May 2023

- chore: prep for v1.0.0 [`#245`](https://github.com/Haidra-Org/hordelib/pull/245) (Jug)

## [v0.19.14](https://github.com/Haidra-Org/hordelib/compare/v0.19.13...v0.19.14)

14 May 2023

- fix: better memory management [`#239`](https://github.com/Haidra-Org/hordelib/pull/239) (Jug)
- fix: remove some pointless dependencies like libcario [`#240`](https://github.com/Haidra-Org/hordelib/pull/240) (Jug)
- Revert "fix: pin timm version to 0.6.13" [`9c82655`](https://github.com/Haidra-Org/hordelib/commit/9c82655ac5f160d8676ade4611c7f157cdde5875)  (Jug)
- feat: adds code to generate all models test page [`b98b992`](https://github.com/Haidra-Org/hordelib/commit/b98b9921600aa77a984812e07c021dc5e6db8e28)  (Jug)
- doc: update readme with all models link [`3478f98`](https://github.com/Haidra-Org/hordelib/commit/3478f984f8eccf9f1b667fe338b5e7349d49c2c9)  (Jug)

## [v0.19.13](https://github.com/Haidra-Org/hordelib/compare/v0.19.12...v0.19.13)

13 May 2023

- fix: better memory management [`#243`](https://github.com/Haidra-Org/hordelib/pull/243) (Jug)

## [v0.19.12](https://github.com/Haidra-Org/hordelib/compare/v0.19.11...v0.19.12)

13 May 2023

- fix: remove some pointless dependencies like libcario [`#240`](https://github.com/Haidra-Org/hordelib/pull/240) (Jug)

## [v0.19.11](https://github.com/Haidra-Org/hordelib/compare/v0.19.10...v0.19.11)

13 May 2023

- fix: pin timm library to v0.6.12 [`0fb0ede`](https://github.com/Haidra-Org/hordelib/commit/0fb0ede022160dd6cd4126e826420b3341b13420)  (Jug)

## [v0.19.10](https://github.com/Haidra-Org/hordelib/compare/v0.19.9...v0.19.10)

12 May 2023

- fix: check underlying model before warm loading from cache [`#236`](https://github.com/Haidra-Org/hordelib/pull/236) (tazlin)
- test: add sampler tests [`#233`](https://github.com/Haidra-Org/hordelib/pull/233) (Jug)
- feat: build a payload to inference time prediction model [`#231`](https://github.com/Haidra-Org/hordelib/pull/231) (Jug)
- fix: pin timm version to 0.6.13 [`2bf710b`](https://github.com/Haidra-Org/hordelib/commit/2bf710b60a803719587aa2cf4af03abab8436a54)  (Jug)
- test: add 10 step sampler tests [`389c6bc`](https://github.com/Haidra-Org/hordelib/commit/389c6bc3d7cf790ccbb3e6d756d5b948a0f6eb52)  (Jug)
- fix: fix kudos model validation [`4805c8b`](https://github.com/Haidra-Org/hordelib/commit/4805c8bf3c4a1c8be98b437c543eb7810d933d4d)  (Jug)

## [v0.19.9](https://github.com/Haidra-Org/hordelib/compare/v0.19.8...v0.19.9)

8 May 2023

- fix: handle image / mask size mismatch [`#229`](https://github.com/Haidra-Org/hordelib/pull/229) (Jug)

## [v0.19.8](https://github.com/Haidra-Org/hordelib/compare/v0.19.7...v0.19.8)

7 May 2023

- fix: faster startup with many models cached [`#224`](https://github.com/Haidra-Org/hordelib/pull/224) (Jug)
- fix: cuts 25+ seconds from load time [`bed3205`](https://github.com/Haidra-Org/hordelib/commit/bed3205984fcc121375fec920d4d551ed9d127c3)  (tazlin)
- fix: updates kudos test [`2c11486`](https://github.com/Haidra-Org/hordelib/commit/2c114869aeac020ba6a7fcd05543dd6ae61c1273)  (Jug)
- hack: disable optimizations for n_iter &gt; 1 [`1a36176`](https://github.com/Haidra-Org/hordelib/commit/1a3617641006bdf632e37e3763311dfb22c1bd9d)  (tazlin)

## [v0.19.7](https://github.com/Haidra-Org/hordelib/compare/v0.19.6...v0.19.7)

3 May 2023

- fix: remove ether real from exclude list [`bd7b082`](https://github.com/Haidra-Org/hordelib/commit/bd7b08213d8c7dcc9bdf47a2e00d35977ffb26ed)  (tazlin)

## [v0.19.6](https://github.com/Haidra-Org/hordelib/compare/v0.19.5...v0.19.6)

3 May 2023

- feat: get model db from legacy model reference repo [`000b8ca`](https://github.com/Haidra-Org/hordelib/commit/000b8ca63a4ba3c773183f8ad46d1fa07d5e67ea)  (tazlin)
- fix: the disaster with linking [`36a605f`](https://github.com/Haidra-Org/hordelib/commit/36a605f42bbda0866245ffa6a5bb6a94fa122572)  (tazlin)
- feat: add a model exclusion list to `consts.py` [`88ee2a6`](https://github.com/Haidra-Org/hordelib/commit/88ee2a6510670d9000ce72a890bdc55e1dad0478)  (tazlin)

## [v0.19.5](https://github.com/Haidra-Org/hordelib/compare/v0.19.4...v0.19.5)

1 May 2023

- fix: remove Ether Real model due to bad hash [`879638b`](https://github.com/Haidra-Org/hordelib/commit/879638b96064e6c185818443c99d64710a3f9daa)  (Jug)
- ci: try upgrading pip before tests [`07f4181`](https://github.com/Haidra-Org/hordelib/commit/07f41815258fddd8cfb642dca34b6eee65b6608d)  (Jug)

## [v0.19.4](https://github.com/Haidra-Org/hordelib/compare/v0.19.3...v0.19.4)

1 May 2023

- fix: update to latest model database [`ef4035c`](https://github.com/Haidra-Org/hordelib/commit/ef4035c00be7caef30a53fff8eabf0fea25662ee)  (Jug)
- fix: fix some model download links [`493c9e1`](https://github.com/Haidra-Org/hordelib/commit/493c9e1968ddae763293c3e0f3d0aaa7368e1004)  (Jug)
- ci: change tests to abort after first failure [`0a51a16`](https://github.com/Haidra-Org/hordelib/commit/0a51a1609a2f46101236baa1edfc560eb25c8469)  (Jug)

## [v0.19.3](https://github.com/Haidra-Org/hordelib/compare/v0.19.2...v0.19.3)

1 May 2023

- ci: update some bits of the release ci [`28d9f29`](https://github.com/Haidra-Org/hordelib/commit/28d9f2979550092249aef82eafcc7f6428a76a25)  (Jug)

## [v0.19.2](https://github.com/Haidra-Org/hordelib/compare/v0.19.1...v0.19.2)

1 May 2023

## [v0.19.1](https://github.com/Haidra-Org/hordelib/compare/v0.19.0...v0.19.1)

1 May 2023

- docs: update readme [`bcc0129`](https://github.com/Haidra-Org/hordelib/commit/bcc0129b6c3f70f42593b47704bcde436db4dc01)  (Jug)

## [v0.19.0](https://github.com/Haidra-Org/hordelib/compare/v0.18.0...v0.19.0)

1 May 2023

## [v0.18.0](https://github.com/Haidra-Org/hordelib/compare/v0.17.0...v0.18.0)

1 May 2023

- feat: use less vram with large images (tiled vae decode) [`#207`](https://github.com/Haidra-Org/hordelib/pull/207) (Jug)
- fix: suppress some clip debug [`ac06694`](https://github.com/Haidra-Org/hordelib/commit/ac06694028cbb58854e9b0dc34b9e90d5c083018)  (Jug)

## [v0.17.0](https://github.com/Haidra-Org/hordelib/compare/v0.16.4...v0.17.0)

1 May 2023

- feat: minor performance tweaking [`#205`](https://github.com/Haidra-Org/hordelib/pull/205) (Jug)
- feat: update model database [`ead7f4a`](https://github.com/Haidra-Org/hordelib/commit/ead7f4a3580873c32ff3f944db11d33958065b1b)  (Jug)
- fix: adds ersgan upscaler, SHA check now case insensitive [`25c4d57`](https://github.com/Haidra-Org/hordelib/commit/25c4d5738f60acd711e285ef71d403183a7628b6)  (tazlin)
- docs: update readme [`3ba2ae0`](https://github.com/Haidra-Org/hordelib/commit/3ba2ae0e435925be24d412015e117334c3af18e6)  (Jug)

## [v0.16.4](https://github.com/Haidra-Org/hordelib/compare/v0.16.3...v0.16.4)

30 April 2023

- fix: support the latest model database format [`d55595a`](https://github.com/Haidra-Org/hordelib/commit/d55595a4f74b17f0e96edd221855d3f50e78b04c)  (Jug)

## [v0.16.3](https://github.com/Haidra-Org/hordelib/compare/v0.16.2...v0.16.3)

29 April 2023

- style: stable_diffusion.json whitespace [`9653cd1`](https://github.com/Haidra-Org/hordelib/commit/9653cd14e64deb781be172f253f293201da3f464)  (tazlin)
- fix: update civitai links out of date, adds two safetensors [`3fe1f3a`](https://github.com/Haidra-Org/hordelib/commit/3fe1f3a0a21823beea024305d22b10d76b547d13)  (tazlin)
- fix: don't allow dynamic prompts to interfere with the random seed. [`e9f29aa`](https://github.com/Haidra-Org/hordelib/commit/e9f29aaee549d13c3b079a6cae41d938232501f8)  (Jug)

## [v0.16.2](https://github.com/Haidra-Org/hordelib/compare/v0.16.1...v0.16.2)

29 April 2023

- fix: ensure we manage ram when loading models from cache [`cce3fba`](https://github.com/Haidra-Org/hordelib/commit/cce3fba6e49138ab390ba1ab8bf35fbbb79128e0)  (Jug)

## [v0.16.1](https://github.com/Haidra-Org/hordelib/compare/v0.16.0...v0.16.1)

29 April 2023

- fix: disk cache model load optimisation [`#198`](https://github.com/Haidra-Org/hordelib/pull/198) (Jug)

## [v0.16.0](https://github.com/Haidra-Org/hordelib/compare/v0.15.3...v0.16.0)

29 April 2023

- feat: automatic resource management [`#186`](https://github.com/Haidra-Org/hordelib/pull/186) (Jug)

## [v0.15.3](https://github.com/Haidra-Org/hordelib/compare/v0.15.2...v0.15.3)

29 April 2023

- feat: add torch and xformers versions to benchmark [`f18630c`](https://github.com/Haidra-Org/hordelib/commit/f18630c62bb95dd21b7d3fee46d2e4a2d1c3c6a4)  (Jug)
- fix: exclude `build/` folder from linting [`66111fb`](https://github.com/Haidra-Org/hordelib/commit/66111fb427c40d44cebe49a55b65a0d9333e02cc)  (tazlin)

## [v0.15.2](https://github.com/Haidra-Org/hordelib/compare/v0.15.1...v0.15.2)

29 April 2023

- fix: validate denoising parameter bounds [`885a190`](https://github.com/Haidra-Org/hordelib/commit/885a190d6d90cd7be08f1747edbf047bd4894e09)  (Jug)
- fix: facefix didn't work on dev versions of torch [`1559f48`](https://github.com/Haidra-Org/hordelib/commit/1559f483484062471193704bc0091613792ab938)  (Jug)
- build: bump to xformers 0.0.19 [`4eab0b9`](https://github.com/Haidra-Org/hordelib/commit/4eab0b9970e4a84539871bcf0cbb29f4d2c795fc)  (Jug)

## [v0.15.1](https://github.com/Haidra-Org/hordelib/compare/v0.15.0...v0.15.1)

27 April 2023

- fix: disable controlnet on low vram gpus in benchmark [`#191`](https://github.com/Haidra-Org/hordelib/pull/191) (Jug)
- fix: rectify txt2img highres denoising [`7757fa1`](https://github.com/Haidra-Org/hordelib/commit/7757fa183d48529ab8509cff09e93246d4e80e2c)  (Jug)

## [v0.15.0](https://github.com/Haidra-Org/hordelib/compare/v0.14.2...v0.15.0)

27 April 2023

- fix: remove unused file [`73bccc8`](https://github.com/Haidra-Org/hordelib/commit/73bccc8ce991cabc503c0b67a979c44ec7c8df2d)  (Jug)
- tests: new test for cuda [`fbc1644`](https://github.com/Haidra-Org/hordelib/commit/fbc164485452f61765fe200cdf334a0ac38725af)  (db0)
- fix: auto fix if width/height not divisible by 64 [`b584952`](https://github.com/Haidra-Org/hordelib/commit/b5849522900a570d8b5ad4b079cc59df476cfa74)  (Jug)

## [v0.14.2](https://github.com/Haidra-Org/hordelib/compare/v0.14.1...v0.14.2)

27 April 2023

- fix: image sizing bugs with hires fix and controlnet [`7e84a65`](https://github.com/Haidra-Org/hordelib/commit/7e84a657fbd9ec716f72fd4d1ddc9a8b052e814c)  (Jug)
- fix: benchmark on linux [`9917b76`](https://github.com/Haidra-Org/hordelib/commit/9917b7623c9b0eb517794c5ca5deecf0414094bd)  (Jug)

## [v0.14.1](https://github.com/Haidra-Org/hordelib/compare/v0.14.0...v0.14.1)

24 April 2023

- fix: use denoising as controlnet strength (compatibility hack) [`#183`](https://github.com/Haidra-Org/hordelib/pull/183) (Jug)

## [v0.14.0](https://github.com/Haidra-Org/hordelib/compare/v0.13.0...v0.14.0)

24 April 2023

- feat: encode prompt pipeline in raw output image metadata [`#181`](https://github.com/Haidra-Org/hordelib/pull/181) (Jug)
- feat: add OS and VRAM to benchmark [`15e065e`](https://github.com/Haidra-Org/hordelib/commit/15e065e49c4f9ffa4805b08441e0f648d534cf60)  (Jug)
- fix: lint fixes [`cdb4da3`](https://github.com/Haidra-Org/hordelib/commit/cdb4da3a3d07a06a734bf87f7f3c85581109c0b1)  (Jug)

## [v0.13.0](https://github.com/Haidra-Org/hordelib/compare/v0.12.1...v0.13.0)

24 April 2023

- feat: adds a hordelib benchmark test [`#179`](https://github.com/Haidra-Org/hordelib/pull/179) (Jug)

## [v0.12.1](https://github.com/Haidra-Org/hordelib/compare/v0.12.0...v0.12.1)

24 April 2023

- fix: unload local models correctly [`0a121e2`](https://github.com/Haidra-Org/hordelib/commit/0a121e2c463f76fc0eef6d35bbfaf30a7381e5a6)  (Jug)
- fix: Clearer logging message for annotator check/download [`d6fed74`](https://github.com/Haidra-Org/hordelib/commit/d6fed7418f6e6b3c52619e6f5c308b9ac6b80e0d)  (tazlin)
- fix: pidinet annotator being downloaded to wrong location [`1bae6c6`](https://github.com/Haidra-Org/hordelib/commit/1bae6c6904ef984182179ba16a413d49c4cf83d5)  (tazlin)

## [v0.12.0](https://github.com/Haidra-Org/hordelib/compare/v0.11.1...v0.12.0)

24 April 2023

- fix: model loaded/unloading stress test fixes [`#175`](https://github.com/Haidra-Org/hordelib/pull/175) (Jug)
- feat: add support for controlnet hires fix [`#173`](https://github.com/Haidra-Org/hordelib/pull/173) (Jug)
- fix: implicitly load local models [`#174`](https://github.com/Haidra-Org/hordelib/pull/174) (Jug)

## [v0.11.1](https://github.com/Haidra-Org/hordelib/compare/v0.11.0...v0.11.1)

23 April 2023

- fix: parameter handling improvements [`#170`](https://github.com/Haidra-Org/hordelib/pull/170) (Jug)

## [v0.11.0](https://github.com/Haidra-Org/hordelib/compare/v0.10.1...v0.11.0)

23 April 2023

- feat: add control_strength parameter for cnet strength [`#167`](https://github.com/Haidra-Org/hordelib/pull/167) (Jug)
- feat: add support for local models including safetensors [`#166`](https://github.com/Haidra-Org/hordelib/pull/166) (Jug)
- feat: upgrade to latest comfyui backend [`#165`](https://github.com/Haidra-Org/hordelib/pull/165) (Jug)

## [v0.10.1](https://github.com/Haidra-Org/hordelib/compare/v0.10.0...v0.10.1)

22 April 2023

- fix: img2img passes 5 thread stress test [`#163`](https://github.com/Haidra-Org/hordelib/pull/163) (Jug)
- fix: unknown samplers and cnets changed to warnings [`325642f`](https://github.com/Haidra-Org/hordelib/commit/325642f9e902bcbde60423a9278c0febb1502549)  (Jug)

## [v0.10.0](https://github.com/Haidra-Org/hordelib/compare/v0.9.5...v0.10.0)

22 April 2023

- feat: add dynamic prompt support [`#161`](https://github.com/Haidra-Org/hordelib/pull/161) (Jug)
- fix: stability fixes [`#159`](https://github.com/Haidra-Org/hordelib/pull/159) (Jug)
- fix: Moves ControlNet Annotators to `AIWORKER_CACHE_HOME` [`63f258d`](https://github.com/Haidra-Org/hordelib/commit/63f258db7fb397a0ada704bb76bd5dd6aeea8761)  (tazlin)
- refactor: cleans up the preload annotators functions [`cee4ddd`](https://github.com/Haidra-Org/hordelib/commit/cee4ddd655cad477ec6ec4751f6e9a9c9b88b69a)  (tazlin)
- feat: Preload controlnet annotators [`9591b15`](https://github.com/Haidra-Org/hordelib/commit/9591b1594ef41805232996cef2db172946061fb1)  (tazlin)

## [v0.9.5](https://github.com/Haidra-Org/hordelib/compare/v0.9.4...v0.9.5)

20 April 2023

- build: fix missing dependency in pypi build [`1610a18`](https://github.com/Haidra-Org/hordelib/commit/1610a185d953d0ee11a27780da82213a0dc6fdfa)  (Jug)

## [v0.9.4](https://github.com/Haidra-Org/hordelib/compare/v0.9.3...v0.9.4)

20 April 2023

- fix: add missing dependency [`fa20f10`](https://github.com/Haidra-Org/hordelib/commit/fa20f1027c0ca133c269704635e1857a3f3114d0)  (Jug)

## [v0.9.3](https://github.com/Haidra-Org/hordelib/compare/v0.9.2...v0.9.3)

20 April 2023

- CI: trigger CI with certain other critical files [`#152`](https://github.com/Haidra-Org/hordelib/pull/152) (tazlin)
- fix: stability fixes [`#150`](https://github.com/Haidra-Org/hordelib/pull/150) (Jug)
- fix: Tox lint/style environments now build (more) correctly [`#151`](https://github.com/Haidra-Org/hordelib/pull/151) (tazlin)
- Revert "Merge branch 'releases' into main" [`80f41c6`](https://github.com/Haidra-Org/hordelib/commit/80f41c645c34d4d0234cec3b1145e11b37b143f4)  (Jug)
- refactor: Housekeeping, preparing for full lint ruleset in CI [`c372d7a`](https://github.com/Haidra-Org/hordelib/commit/c372d7ad4f565f25ba99b44ee117f1d09229d5c5)  (tazlin)
- refactor: Control net model manager housekeeping [`8c600fd`](https://github.com/Haidra-Org/hordelib/commit/8c600fd3ed1fa1bb341f72f9b737bd46717a97c2)  (tazlin)

## [v0.9.2](https://github.com/Haidra-Org/hordelib/compare/v0.9.1...v0.9.2)

17 April 2023

- fix: don't mix up controlnets and run out of vram [`#147`](https://github.com/Haidra-Org/hordelib/pull/147) (Jug)

## [v0.9.1](https://github.com/Haidra-Org/hordelib/compare/v0.9.0...v0.9.1)

17 April 2023

- fix: add proper exception logging to comfyui, closes #64 [`#64`](https://github.com/Haidra-Org/hordelib/issues/64)  ()

## [v0.9.0](https://github.com/Haidra-Org/hordelib/compare/v0.8.8...v0.9.0)

16 April 2023

- feat: active memory and model management [`#144`](https://github.com/Haidra-Org/hordelib/pull/144) (Jug)

## [v0.8.8](https://github.com/Haidra-Org/hordelib/compare/v0.8.7...v0.8.8)

15 April 2023

- fix: Make thread locking as minimalist as possible [`#142`](https://github.com/Haidra-Org/hordelib/pull/142) (Jug)
- fix: fix broken stress test [`a713524`](https://github.com/Haidra-Org/hordelib/commit/a713524f8a3511ed28e86f240e13bc17cb7b51c4)  (Jug)

## [v0.8.7](https://github.com/Haidra-Org/hordelib/compare/v0.8.6...v0.8.7)

15 April 2023

- fix: don't thread lock loading with inference [`f9e4d2c`](https://github.com/Haidra-Org/hordelib/commit/f9e4d2c29ea9efb10c62b687e1c720dd8e1a5a3b)  (Jug)
- chore: more badge refresh tweaks [`58b6902`](https://github.com/Haidra-Org/hordelib/commit/58b6902b5e852a080e51e8830b31560503c4791d)  (Jug)

## [v0.8.6](https://github.com/Haidra-Org/hordelib/compare/v0.8.5...v0.8.6)

15 April 2023

- fix: Sha validation fix [`#139`](https://github.com/Haidra-Org/hordelib/pull/139) (tazlin)
- fix: pytest discovery, broken by non-tests in test folder [`6c81986`](https://github.com/Haidra-Org/hordelib/commit/6c8198671924bd8edef30c15dc34dffd477bd49d)  (tazlin)
- fix: switches pr CI to use example/ run_* [`fa3795c`](https://github.com/Haidra-Org/hordelib/commit/fa3795c6307e7af26a663986a844be6a24f954f9)  (tazlin)
- build: update CI to do a weak lint/formatting check [`da4acc1`](https://github.com/Haidra-Org/hordelib/commit/da4acc1f055b4bd60dc181e555181ca94e080e92)  (tazlin)

## [v0.8.5](https://github.com/Haidra-Org/hordelib/compare/v0.8.4...v0.8.5)

14 April 2023

- test: add threaded torture test [`6ef872c`](https://github.com/Haidra-Org/hordelib/commit/6ef872c2b60f5441b7bd4a379ad3402a25b8a21e)  (Jug)
- fix: assert parameter bounds to stop errors [`1af9726`](https://github.com/Haidra-Org/hordelib/commit/1af972683eb051f4dd6ce04afff6cf1925a92d8b)  (Jug)

## [v0.8.4](https://github.com/Haidra-Org/hordelib/compare/v0.8.3...v0.8.4)

14 April 2023

- fix: threading and job settings being mixed together [`#127`](https://github.com/Haidra-Org/hordelib/pull/127) (Jug)
- ci: try to refresh pypi badge on release [`ad59dea`](https://github.com/Haidra-Org/hordelib/commit/ad59dea74dea25cb94d30a60f91d7ad90d09abf0)  (Jug)
- docs: minor url tweak in readme [`38d64b5`](https://github.com/Haidra-Org/hordelib/commit/38d64b5c6579714a02090df4960ca4aaa5646a0c)  (Jug)

## [v0.8.3](https://github.com/Haidra-Org/hordelib/compare/v0.8.2...v0.8.3)

13 April 2023

- fix: defer model manager loading [`8a7520b`](https://github.com/Haidra-Org/hordelib/commit/8a7520bc32db6f7d84dfe650a2f67adba26a49bb)  (tazlin)

## [v0.8.2](https://github.com/Haidra-Org/hordelib/compare/v0.8.1...v0.8.2)

13 April 2023

- feat: performance optimisation [`#125`](https://github.com/Haidra-Org/hordelib/pull/125) (Jug)
- refactor: Logger tweaks, Model Manager housekeeping [`#118`](https://github.com/Haidra-Org/hordelib/pull/118) (tazlin)
- docs: Update README.md [`8d7397e`](https://github.com/Haidra-Org/hordelib/commit/8d7397e4b50187c9b4f9ce2d67bbbdd851069893)  (tazlin)
- ci: remove junk from changelog [`bc755b3`](https://github.com/Haidra-Org/hordelib/commit/bc755b38995752606058d90634dbe13dbb592067)  (Jug)

## [v0.8.1](https://github.com/Haidra-Org/hordelib/compare/v0.8.0...v0.8.1)

13 April 2023

- fix: suppress terminal spam from comfyui [`45457ed`](https://github.com/Haidra-Org/hordelib/commit/45457ed566b383dcb36aa49d46faa1e2d20ca5fd)  (Jug)

## [v0.8.0](https://github.com/Haidra-Org/hordelib/compare/v0.7.3...v0.8.0)

13 April 2023

- feat: Clip Rankings [`#117`](https://github.com/Haidra-Org/hordelib/pull/117) (Divided by Zer0)
- feat: Blip [`#116`](https://github.com/Haidra-Org/hordelib/pull/116) (Divided by Zer0)
- fix: make library thread safe [`abc8074`](https://github.com/Haidra-Org/hordelib/commit/abc80743401901b1da46b3bbc8b4221099590a52)  (Jug)
- fix: remove thread mutex for now [`bccc81b`](https://github.com/Haidra-Org/hordelib/commit/bccc81b22a20725bfe86921c7cd4288f244351a6)  (Jug)
- build: fix build_helper for local use [`835a7ca`](https://github.com/Haidra-Org/hordelib/commit/835a7cab94506b0daa4dd757d827a2cd416f4c3d)  (Jug)

## [v0.7.3](https://github.com/Haidra-Org/hordelib/compare/v0.7.2...v0.7.3)

12 April 2023

- build: more production build fixes. [`cf8d3be`](https://github.com/Haidra-Org/hordelib/commit/cf8d3be76f7466fe9931720810792c0accf6bc12)  (Jug)

## [v0.7.2](https://github.com/Haidra-Org/hordelib/compare/v0.7.1...v0.7.2)

12 April 2023

- build: fix production build packaging [`fc9da95`](https://github.com/Haidra-Org/hordelib/commit/fc9da958e471d6d8baa4638bb1ec38d018f78bd4)  (Jug)

## [v0.7.1](https://github.com/Haidra-Org/hordelib/compare/v0.7.0...v0.7.1)

12 April 2023

- build: fix missing build time dependency [`57292fc`](https://github.com/Haidra-Org/hordelib/commit/57292fc46ef5a8f2d28eba4158829dfd69868c53)  (Jug)

## [v0.7.0](https://github.com/Haidra-Org/hordelib/compare/v0.6.1...v0.7.0)

12 April 2023

- build: add support for production builld [`#109`](https://github.com/Haidra-Org/hordelib/pull/109) (Jug)
- build: fix detection of production build [`131ee91`](https://github.com/Haidra-Org/hordelib/commit/131ee91df9206e5804b78fad64d11ae278798716)  (Jug)
- ci: if a label is forgotten on release, assume patch release [`7d05741`](https://github.com/Haidra-Org/hordelib/commit/7d05741557a48b0648493da0b84672815e76e2b3)  (Jug)

## [v0.6.1](https://github.com/Haidra-Org/hordelib/compare/v0.6.0...v0.6.1)

12 April 2023

- feat: make logging setup and control optional [`#106`](https://github.com/Haidra-Org/hordelib/pull/106) (Jug)
- style: Automatic formatting/lint with length 119 [`0e9375a`](https://github.com/Haidra-Org/hordelib/commit/0e9375af76f00284d367def57f8515ac3ecc9f11)  (tazlin)
- docs: cleanup readme for viewing in an editor [`700e23e`](https://github.com/Haidra-Org/hordelib/commit/700e23ea9ffdf57a71625efac2dede6da56d8d6a)  (Jug)
- chore: Change black to line length 119 [`7b27d81`](https://github.com/Haidra-Org/hordelib/commit/7b27d812b79374eed6c2f7751cc9084afca4a451)  (tazlin)

## [v0.6.0](https://github.com/Haidra-Org/hordelib/compare/v0.5.24...v0.6.0)

12 April 2023

- fix: suppress terminal spam [`#104`](https://github.com/Haidra-Org/hordelib/pull/104) (Jug)
- feat: add support for separate source_mask [`#103`](https://github.com/Haidra-Org/hordelib/pull/103) (Jug)

## [v0.5.24](https://github.com/Haidra-Org/hordelib/compare/v0.5.23...v0.5.24)

12 April 2023

- ci: include changelog link on pypi page [`69a7a0d`](https://github.com/Haidra-Org/hordelib/commit/69a7a0d06e3abdce6ede565ecbfd52fc4fa973b1)  (Jug)

## [v0.5.23](https://github.com/Haidra-Org/hordelib/compare/v0.5.22...v0.5.23)

12 April 2023

- ci: customise the changelog format [`1d9de8d`](https://github.com/Haidra-Org/hordelib/commit/1d9de8dab12004bcd842a5c7ad372a25bf8869c9)  (Jug)

## [v0.5.22](https://github.com/Haidra-Org/hordelib/compare/v0.5.21...v0.5.22)

12 April 2023

## [v0.5.21](https://github.com/Haidra-Org/hordelib/compare/v0.5.20...v0.5.21)

12 April 2023

- ci: try to generate changelog for the right version [`649678a`](https://github.com/Haidra-Org/hordelib/commit/649678a4429ffbda0151f1ef4537bf1b4801dda5)  (Jug)

## [v0.5.20](https://github.com/Haidra-Org/hordelib/compare/v0.5.19...v0.5.20)

12 April 2023

- ci: try a better changelog generator [`c741bb8`](https://github.com/Haidra-Org/hordelib/commit/c741bb8118dd7bca1e07f665e957a59ba0473f08)  (Jug)
- ci: more tweaks [`98c9409`](https://github.com/Haidra-Org/hordelib/commit/98c9409fea08c92e91664e3ddd2a3f458c30f20d)  (Jug)
- ci: ci again [`32105cd`](https://github.com/Haidra-Org/hordelib/commit/32105cd75da321355eb30c40c6f1c00d00e13248)  (Jug)

## [v0.5.19](https://github.com/Haidra-Org/hordelib/compare/v0.5.18...v0.5.19)

12 April 2023

- ci: changelog wasn't include in setuptools [`b6a6175`](https://github.com/Haidra-Org/hordelib/commit/b6a61754d7d9877e28a9383aec9b2a2ddc7a20df)  (Jug)

## [v0.5.18](https://github.com/Haidra-Org/hordelib/compare/v0.5.17...v0.5.18)

11 April 2023

- ci: release ci tweaks [`bae4e00`](https://github.com/Haidra-Org/hordelib/commit/bae4e005b995390959aed41decc0de5d8165aa23)  (Jug)

## [v0.5.17](https://github.com/Haidra-Org/hordelib/compare/v0.5.16...v0.5.17)

11 April 2023

- style: Incremental style/lint catchup [`d2ec8c3`](https://github.com/Haidra-Org/hordelib/commit/d2ec8c34e08e58a7cabf13777734cdbf881eea01)  (tazlin)
- ci: Enables a couple ruff rules for CI [`b4ebac6`](https://github.com/Haidra-Org/hordelib/commit/b4ebac60f606244db13dd12ca77a36bde4b4fa8c)  (tazlin)
- ci: another day another way to do changelogs [`51b9df7`](https://github.com/Haidra-Org/hordelib/commit/51b9df735b701d1bb31b2f92c56051d33cd38547)  (Jug)

## [v0.5.16](https://github.com/Haidra-Org/hordelib/compare/v0.5.15...v0.5.16)

11 April 2023

- ci: more ci tweaks [`54f3ab8`](https://github.com/Haidra-Org/hordelib/commit/54f3ab876f7e19c3ae29271b4c94326f97d26372)  (Jug)

## [v0.5.15](https://github.com/Haidra-Org/hordelib/compare/v0.5.14...v0.5.15)

11 April 2023

- ci: this is never going to work is it [`0254cc8`](https://github.com/Haidra-Org/hordelib/commit/0254cc821ec455d69ed6c5926dbc499471330062)  (Jug)

## [v0.5.14](https://github.com/Haidra-Org/hordelib/compare/v0.5.13...v0.5.14)

11 April 2023

- ci: another day another ci hack [`1a6d47e`](https://github.com/Haidra-Org/hordelib/commit/1a6d47e10c8d219702b691a11cdc632db45cf45d)  (Jug)

## [v0.5.13](https://github.com/Haidra-Org/hordelib/compare/v0.5.12...v0.5.13)

11 April 2023

- ci: optimistically try to output a changelog [`30c82b7`](https://github.com/Haidra-Org/hordelib/commit/30c82b7577654488658036f18bf21cd03c415a6b)  (Jug)

## [v0.5.12](https://github.com/Haidra-Org/hordelib/compare/v0.5.11...v0.5.12)

11 April 2023

- ci: Try harder to generate a changelog [`6e0f68f`](https://github.com/Haidra-Org/hordelib/commit/6e0f68f48d67783f9668b886572487f9fb11a233)  (Jug)

## [v0.5.11](https://github.com/Haidra-Org/hordelib/compare/v0.5.10...v0.5.11)

11 April 2023

- ci: add some notes to the release ci [`e89d255`](https://github.com/Haidra-Org/hordelib/commit/e89d25520e86d6a683daf7fc9b6451f1a53f37cd)  (Jug)
- ci: generate a changelog once again [`d250b60`](https://github.com/Haidra-Org/hordelib/commit/d250b6072bad0081ff779d31db0017603bee201b)  (Jug)

## [v0.5.10](https://github.com/Haidra-Org/hordelib/compare/v0.5.9...v0.5.10)

11 April 2023

- docs: remove changelog [`325f02d`](https://github.com/Haidra-Org/hordelib/commit/325f02d3a7cd9020392fde10582a9934eaf66f31)  (Jug)
- ci: tweak release scripts [`818667a`](https://github.com/Haidra-Org/hordelib/commit/818667a1a39858ebe536063d93d747cbc8928dda)  (Jug)

## [v0.5.9](https://github.com/Haidra-Org/hordelib/compare/v0.5.8...v0.5.9)

11 April 2023

- ci: more tweaks to the ci process [`60dc19d`](https://github.com/Haidra-Org/hordelib/commit/60dc19db63cb2c9178aae9a7caac51a65c73f833)  (Jug)

## [v0.5.8](https://github.com/Haidra-Org/hordelib/compare/v0.5.7...v0.5.8)

11 April 2023

## [v0.5.7](https://github.com/Haidra-Org/hordelib/compare/v0.5.6...v0.5.7)

11 April 2023

- fix: img2img + highres_fix  [`#80`](https://github.com/Haidra-Org/hordelib/pull/80) (Divided by Zer0)
- ci: try to publish to pypi on release [`d95f14e`](https://github.com/Haidra-Org/hordelib/commit/d95f14eb867395dfee2e608e8319dfa70540d077)  (Jug)
- ci: tweaks to the release ci [`56e1fe7`](https://github.com/Haidra-Org/hordelib/commit/56e1fe7ef109df43bdc5f708543d496aa3cf82dc)  (Jug)

## [v0.5.6](https://github.com/Haidra-Org/hordelib/compare/v0.5.5...v0.5.6)

11 April 2023

- tests: class scope on inference tests for speedup [`#78`](https://github.com/Haidra-Org/hordelib/pull/78) (Divided by Zer0)
- docs: recreate LICENSE [`acb3c10`](https://github.com/Haidra-Org/hordelib/commit/acb3c10cb9c888eecf42cddc39d7e097b8fcb534)  (Jug)
- docs: remove license to recreate it [`0b1b67f`](https://github.com/Haidra-Org/hordelib/commit/0b1b67f4b8264b862079f9c8fc440b3e9bca49e3)  (Jug)
- build: placeholder changelog [`d9ace8c`](https://github.com/Haidra-Org/hordelib/commit/d9ace8cf5403cb29f0c879e289d3967170fc1ace)  (Jug)

## [v0.5.5](https://github.com/Haidra-Org/hordelib/compare/v0.5.4...v0.5.5)

11 April 2023

## [v0.5.4](https://github.com/Haidra-Org/hordelib/compare/v0.5.3...v0.5.4)

11 April 2023

- build: add release mode flag [`#76`](https://github.com/Haidra-Org/hordelib/pull/76) (Jug)
- refactor!: Second big Model Manager rework step [`#75`](https://github.com/Haidra-Org/hordelib/pull/75) (tazlin)
- fix: adjust mlsd annotator defaults [`#74`](https://github.com/Haidra-Org/hordelib/pull/74) (Jug)
- chore: resolve merge conflicts [`9de7f92`](https://github.com/Haidra-Org/hordelib/commit/9de7f92886065c799449831e6809089bac9d98c5)  (Jug)
- docs: remove the changelog from main [`7cfdb60`](https://github.com/Haidra-Org/hordelib/commit/7cfdb601f6f82c7e8592abaf57ce13e0ea1cb5b7)  (Jug)
- fix: normal map and mlsd annotators [`bb7175d`](https://github.com/Haidra-Org/hordelib/commit/bb7175df92fbc244dcac168b7e5b755e7d8785c2)  (Jug)

## [v0.5.3](https://github.com/Haidra-Org/hordelib/compare/v0.5.2...v0.5.3)

11 April 2023

- build: patch release [`#73`](https://github.com/Haidra-Org/hordelib/pull/73) (Jug)
- build: try to fix test running and build [`9dab8ef`](https://github.com/Haidra-Org/hordelib/commit/9dab8ef9dd85e158796e907b4aab7822dfe3d931)  (Jug)

## [v0.5.2](https://github.com/Haidra-Org/hordelib/compare/v0.5.1...v0.5.2)

11 April 2023

- build: upgrade to torch 2, xformers 18 and latest comfyui [`#68`](https://github.com/Haidra-Org/hordelib/pull/68) (Jug)

## [v0.5.1](https://github.com/Haidra-Org/hordelib/compare/v0.5.0...v0.5.1)

11 April 2023

- feat: Added is_model_loaded() to HyperMM [`#67`](https://github.com/Haidra-Org/hordelib/pull/67) (Divided by Zer0)

## [v0.5.0](https://github.com/Haidra-Org/hordelib/compare/v0.4.2...v0.5.0)

11 April 2023

- feat: add support for return_control_map [`#66`](https://github.com/Haidra-Org/hordelib/pull/66) (Jug)
- docs: update ci test badge [`68ad0cb`](https://github.com/Haidra-Org/hordelib/commit/68ad0cb0947f1d6cd30aff59542ebb41fcb29445)  (Jug)

## [v0.4.2](https://github.com/Haidra-Org/hordelib/compare/v0.4.1...v0.4.2)

11 April 2023

- fix: resize img2img before inference [`#63`](https://github.com/Haidra-Org/hordelib/pull/63) (Divided by Zer0)
- fix: add timezone to build results [`#61`](https://github.com/Haidra-Org/hordelib/pull/61) (Jug)
- tests: gfpgan test and size assets [`#62`](https://github.com/Haidra-Org/hordelib/pull/62) (Divided by Zer0)
- docs: update with pypi test notes [`183e352`](https://github.com/Haidra-Org/hordelib/commit/183e352cae0bf8f44fee8d7f162789a48ca1b670)  (Jug)

## [v0.4.1](https://github.com/Haidra-Org/hordelib/compare/v0.4.0...v0.4.1)

10 April 2023

- feat: Make use of the ControlNet ModelManager [`#53`](https://github.com/Haidra-Org/hordelib/pull/53) (Divided by Zer0)
- test: fix test with red border around it [`#58`](https://github.com/Haidra-Org/hordelib/pull/58) (Jug)
- build: activate build results website [`#57`](https://github.com/Haidra-Org/hordelib/pull/57) (Jug)
- build: make a webpage of test result images [`#55`](https://github.com/Haidra-Org/hordelib/pull/55) (Jug)
- test: fix black 64x64 image tests [`#54`](https://github.com/Haidra-Org/hordelib/pull/54) (Jug)
- version incremented [`6508542`](https://github.com/Haidra-Org/hordelib/commit/6508542bab6175f4fc353da417098e29b66cb90e)  (github-actions)
- build: try to be smarter when we run tests [`221d415`](https://github.com/Haidra-Org/hordelib/commit/221d415e08ee0942108e65822b6a51a9da6648a2)  (Jug)
- build: try to run tests more often [`e7a63d8`](https://github.com/Haidra-Org/hordelib/commit/e7a63d8e4a8ce8ce1b6918fcd4076669c7ffa9cd)  (Jug)

## [v0.4.0](https://github.com/Haidra-Org/hordelib/compare/v0.3.1...v0.4.0)

10 April 2023

- feat: add face fixing support [`#50`](https://github.com/Haidra-Org/hordelib/pull/50) (Jug)
- version incremented [`5291422`](https://github.com/Haidra-Org/hordelib/commit/5291422c3f4ca7e986b798be3946fba67609e4f4)  (github-actions)

## [v0.3.1](https://github.com/Haidra-Org/hordelib/compare/v0.3.0...v0.3.1)

10 April 2023

- test: change all tests to webp [`#49`](https://github.com/Haidra-Org/hordelib/pull/49) (Jug)
- version incremented [`22e594c`](https://github.com/Haidra-Org/hordelib/commit/22e594cb177ed2d18e3094d868d0a8d07400725c)  (github-actions)

## [v0.3.0](https://github.com/Haidra-Org/hordelib/compare/v0.2.2...v0.3.0)

10 April 2023

- feat: add controlnet support [`#46`](https://github.com/Haidra-Org/hordelib/pull/46) (Jug)
- version incremented [`ce239bb`](https://github.com/Haidra-Org/hordelib/commit/ce239bbe073798476a503ddd7d7c9ae23319c378)  (github-actions)
- docs: add build status badge to readme [`637a7c4`](https://github.com/Haidra-Org/hordelib/commit/637a7c4d1bf90b07d986f272b4f1b0fc914941ff)  (Jug)

## [v0.2.2](https://github.com/Haidra-Org/hordelib/compare/v0.2.1...v0.2.2)

9 April 2023

- ci: inpainting tests [`#47`](https://github.com/Haidra-Org/hordelib/pull/47) (Divided by Zer0)
- version incremented [`43d827e`](https://github.com/Haidra-Org/hordelib/commit/43d827e05f7512b331b9e75e18252c9eb124fbcc)  (github-actions)

## [v0.2.1](https://github.com/Haidra-Org/hordelib/compare/v0.2.0...v0.2.1)

9 April 2023

- build: change how custom nodes are loaded into comfyui [`#44`](https://github.com/Haidra-Org/hordelib/pull/44) (Jug)

## [v0.2.0](https://github.com/Haidra-Org/hordelib/compare/v0.1.0...v0.2.0)

9 April 2023

- ci: Disable pypi publish [`#45`](https://github.com/Haidra-Org/hordelib/pull/45) (Divided by Zer0)
- docs: readme updates. [`#43`](https://github.com/Haidra-Org/hordelib/pull/43) (Jug)
- docs: readme updates. [`#42`](https://github.com/Haidra-Org/hordelib/pull/42) (Jug)
- feat: Re-adds diffusers model manager [`#41`](https://github.com/Haidra-Org/hordelib/pull/41) (tazlin)
- test: add diffusers inpainting run example [`#40`](https://github.com/Haidra-Org/hordelib/pull/40) (Jug)
- docs: update readme [`#39`](https://github.com/Haidra-Org/hordelib/pull/39) (Jug)
- refactor: We do some light refactoring... [`#34`](https://github.com/Haidra-Org/hordelib/pull/34) (Divided by Zer0)
- test: Optimized tests [`#32`](https://github.com/Haidra-Org/hordelib/pull/32) (Divided by Zer0)
- refactor: Significant code cleanup and CI/build improvements. [`#30`](https://github.com/Haidra-Org/hordelib/pull/30) (tazlin)
- feat: Post processors [`#27`](https://github.com/Haidra-Org/hordelib/pull/27) (Divided by Zer0)
- feat: adds inpainting [`1cae790`](https://github.com/Haidra-Org/hordelib/commit/1cae7908b387753b6dfb9475270ccc3ace83a0cf)  (Jug)
- feat: image loader, basic img2img [`c9ba83b`](https://github.com/Haidra-Org/hordelib/commit/c9ba83bc7606f1df005cc2541fc199a3821a53eb)  (Jug)
- test: reduce vram requirements for hires fix tests [`7ced31a`](https://github.com/Haidra-Org/hordelib/commit/7ced31aa2f8f0463d982aed1d80152f21de378db)  (Jug)

## [v0.1.0](https://github.com/Haidra-Org/hordelib/compare/v0.0.10...v0.1.0)

6 April 2023

- fix: Duplicate lines [`#25`](https://github.com/Haidra-Org/hordelib/pull/25) (tazlin)
- feat: Adds a github action when pushing to main that will generate a new release and an automatic changelog [`#24`](https://github.com/Haidra-Org/hordelib/pull/24) (Jug)
- fix: References to `horde_model_manager`, more docs [`#23`](https://github.com/Haidra-Org/hordelib/pull/23) (tazlin)
- docs: Update LICENSE [`#20`](https://github.com/Haidra-Org/hordelib/pull/20) (tazlin)
- refactor: ModelManager improvements, test adjustments [`#19`](https://github.com/Haidra-Org/hordelib/pull/19) (tazlin)
- fix: missing return [`#18`](https://github.com/Haidra-Org/hordelib/pull/18) (Divided by Zer0)
- refactor: 'ModelManager' rework, added 'WorkerSettings' [`#17`](https://github.com/Haidra-Org/hordelib/pull/17) (tazlin)
- refactor: Test tweaks, type hint fixes [`#16`](https://github.com/Haidra-Org/hordelib/pull/16) (tazlin)
- refactor: Type hints, refactoring, preemptive checks [`#15`](https://github.com/Haidra-Org/hordelib/pull/15) (tazlin)
- feat: adds clip skip support [`64f0a59`](https://github.com/Haidra-Org/hordelib/commit/64f0a59d019fd6b256bebe58efd5285f67da2c17)  (Jug)
- refactor: make things more explicit. [`e90ca84`](https://github.com/Haidra-Org/hordelib/commit/e90ca841e96d62d3137db88ea3ba117a498f3e38)  (Jug)
- feat: allow running comfyui easily. [`d3823b3`](https://github.com/Haidra-Org/hordelib/commit/d3823b38d5905be4f5bad1461831daac32c64337)  (Jug)

## [v0.0.10](https://github.com/Haidra-Org/hordelib/compare/v0.0.9...v0.0.10)

3 April 2023

- fix: test_horde.py syntax error [`#14`](https://github.com/Haidra-Org/hordelib/pull/14) (tazlin)
- fix: Compat fixes for tests from pr #11 [`#12`](https://github.com/Haidra-Org/hordelib/pull/12) (tazlin)
- feat: Clip interrogation support [`#11`](https://github.com/Haidra-Org/hordelib/pull/11) (tazlin)
- feat: Adds support for using a Model Manager  [`#8`](https://github.com/Haidra-Org/hordelib/pull/8) (Divided by Zer0)
- build: fixes for new model manager and clip [`2984e8b`](https://github.com/Haidra-Org/hordelib/commit/2984e8bb71e33285c89ef01ff80666411a46554d)  (Jug)
- build: update comfyui to latest version [`a5cfc05`](https://github.com/Haidra-Org/hordelib/commit/a5cfc05352f8a86c9beb511ab5b869a2c72b6cb3)  (Jug)
- build: disable forced reformatting from black [`2a33a1f`](https://github.com/Haidra-Org/hordelib/commit/2a33a1ff2610edda766a5c4d5c7680fdc7c97237)  (Jug)

## [v0.0.9](https://github.com/Haidra-Org/hordelib/compare/v0.0.8...v0.0.9)

3 April 2023

- test: More thorough tests for ComfyUI install [`a247f2b`](https://github.com/Haidra-Org/hordelib/commit/a247f2b9fd9b62c3b0718468a50859e06c92ee65)  (Jug)
- build: adds build helper script. [`afd38ea`](https://github.com/Haidra-Org/hordelib/commit/afd38eabbf103b52c32a8e4daff2e7f1e7b2324d)  (Jug)
- build: don't run inference tests on github (no cuda) [`638056b`](https://github.com/Haidra-Org/hordelib/commit/638056bc2d48098a4cdcad592ab955abe815fcd7)  (Jug)

## [v0.0.8](https://github.com/Haidra-Org/hordelib/compare/v0.0.1...v0.0.8)

2 April 2023

## v0.0.1

4 June 2023

- fix: tighten up thread safety around the sampler [`#341`](https://github.com/Haidra-Org/hordelib/pull/341) (Jug)
- fix: don't return results of another job in certain corner cases. [`#339`](https://github.com/Haidra-Org/hordelib/pull/339) (Jug)
- fix: logging error with loading cnet [`#332`](https://github.com/Haidra-Org/hordelib/pull/332) (Jug)
- fix: allow all types of downloads to display progress [`#324`](https://github.com/Haidra-Org/hordelib/pull/324) (Jug)
- fix: make index uses .png files [`#321`](https://github.com/Haidra-Org/hordelib/pull/321) (Divided by Zer0)
- tests: Compare test images with expected output [`#319`](https://github.com/Haidra-Org/hordelib/pull/319) (Divided by Zer0)
- feat: add support for download progress indicators [`#318`](https://github.com/Haidra-Org/hordelib/pull/318) (Jug)
- fix: make fakescribble controlnet work again [`#314`](https://github.com/Haidra-Org/hordelib/pull/314) (Jug)
- fix: hangs and random processing results with multiple threads regression [`#311`](https://github.com/Haidra-Org/hordelib/pull/311) (Jug)
- fix: ensure lora folder exists before starting download [`#309`](https://github.com/Haidra-Org/hordelib/pull/309) (Divided by Zer0)
- fix: more robust downloads; resume, retry, don't delete files so hastily.  [`#307`](https://github.com/Haidra-Org/hordelib/pull/307) (Jug)
- fix: moved lora downloads outside of init [`#304`](https://github.com/Haidra-Org/hordelib/pull/304) (Divided by Zer0)
- Lora Model Manager [`#302`](https://github.com/Haidra-Org/hordelib/pull/302) (Divided by Zer0)
- fix: fix some tests and update docs for Linux [`#301`](https://github.com/Haidra-Org/hordelib/pull/301) (Jug)
- fix: remove "No job ran for x seconds" warning [`#298`](https://github.com/Haidra-Org/hordelib/pull/298) (Jug)
- fix: ignore unknown loras, search case insensitively for them [`#297`](https://github.com/Haidra-Org/hordelib/pull/297) (Jug)
- fix: unit tests use about 6GB VRAM max now. [`#293`](https://github.com/Haidra-Org/hordelib/pull/293) (Jug)
- feat: refactor for clarity, tweak img2img and inpainting, tidy tests [`#290`](https://github.com/Haidra-Org/hordelib/pull/290) (Jug)
- Add alt pipeline design for img2img with mask [`#279`](https://github.com/Haidra-Org/hordelib/pull/279) (Wolfgang Meyers)
- fix: correctly output pipeline json during development [`#284`](https://github.com/Haidra-Org/hordelib/pull/284) (Jug)
- fix: auto fix bad cfg values [`#282`](https://github.com/Haidra-Org/hordelib/pull/282) (Jug)
- feat: add lora support and reduce cnet memory requirements by 50% [`#270`](https://github.com/Haidra-Org/hordelib/pull/270) (Jug)
- Increase read/write sizes during download/checksums [`#274`](https://github.com/Haidra-Org/hordelib/pull/274) (Andy Pilate)
- When gathering loaded/available names, allows filtering by model manager type [`#254`](https://github.com/Haidra-Org/hordelib/pull/254) (Divided by Zer0)
- feat: upgrade to the latest comfyui [`#255`](https://github.com/Haidra-Org/hordelib/pull/255) (Jug)
- feat: add option to enable/disable batch optimisation [`#252`](https://github.com/Haidra-Org/hordelib/pull/252) (Jug)
- fix: correctly unload models from gpu under stress [`#249`](https://github.com/Haidra-Org/hordelib/pull/249) (Jug)
- fix: benchmark looks harder for model directory [`#247`](https://github.com/Haidra-Org/hordelib/pull/247) (Jug)
- chore: prep for v1.0.0 [`#245`](https://github.com/Haidra-Org/hordelib/pull/245) (Jug)
- fix: better memory management [`#239`](https://github.com/Haidra-Org/hordelib/pull/239) (Jug)
- fix: remove some pointless dependencies like libcario [`#240`](https://github.com/Haidra-Org/hordelib/pull/240) (Jug)
- fix: remove some pointless dependencies like libcario [`#240`](https://github.com/Haidra-Org/hordelib/pull/240) (Jug)
- fix: check underlying model before warm loading from cache [`#236`](https://github.com/Haidra-Org/hordelib/pull/236) (tazlin)
- test: add sampler tests [`#233`](https://github.com/Haidra-Org/hordelib/pull/233) (Jug)
- feat: build a payload to inference time prediction model [`#231`](https://github.com/Haidra-Org/hordelib/pull/231) (Jug)
- fix: handle image / mask size mismatch [`#229`](https://github.com/Haidra-Org/hordelib/pull/229) (Jug)
- fix: faster startup with many models cached [`#224`](https://github.com/Haidra-Org/hordelib/pull/224) (Jug)
- feat: use less vram with large images (tiled vae decode) [`#207`](https://github.com/Haidra-Org/hordelib/pull/207) (Jug)
- feat: minor performance tweaking [`#205`](https://github.com/Haidra-Org/hordelib/pull/205) (Jug)
- fix: disk cache model load optimisation [`#198`](https://github.com/Haidra-Org/hordelib/pull/198) (Jug)
- feat: automatic resource management [`#186`](https://github.com/Haidra-Org/hordelib/pull/186) (Jug)
- fix: disable controlnet on low vram gpus in benchmark [`#191`](https://github.com/Haidra-Org/hordelib/pull/191) (Jug)
- fix: use denoising as controlnet strength (compatibility hack) [`#183`](https://github.com/Haidra-Org/hordelib/pull/183) (Jug)
- feat: encode prompt pipeline in raw output image metadata [`#181`](https://github.com/Haidra-Org/hordelib/pull/181) (Jug)
- feat: adds a hordelib benchmark test [`#179`](https://github.com/Haidra-Org/hordelib/pull/179) (Jug)
- fix: model loaded/unloading stress test fixes [`#175`](https://github.com/Haidra-Org/hordelib/pull/175) (Jug)
- feat: add support for controlnet hires fix [`#173`](https://github.com/Haidra-Org/hordelib/pull/173) (Jug)
- fix: implicitly load local models [`#174`](https://github.com/Haidra-Org/hordelib/pull/174) (Jug)
- fix: parameter handling improvements [`#170`](https://github.com/Haidra-Org/hordelib/pull/170) (Jug)
- feat: add control_strength parameter for cnet strength [`#167`](https://github.com/Haidra-Org/hordelib/pull/167) (Jug)
- feat: add support for local models including safetensors [`#166`](https://github.com/Haidra-Org/hordelib/pull/166) (Jug)
- feat: upgrade to latest comfyui backend [`#165`](https://github.com/Haidra-Org/hordelib/pull/165) (Jug)
- fix: img2img passes 5 thread stress test [`#163`](https://github.com/Haidra-Org/hordelib/pull/163) (Jug)
- feat: add dynamic prompt support [`#161`](https://github.com/Haidra-Org/hordelib/pull/161) (Jug)
- fix: stability fixes [`#159`](https://github.com/Haidra-Org/hordelib/pull/159) (Jug)
- CI: trigger CI with certain other critical files [`#152`](https://github.com/Haidra-Org/hordelib/pull/152) (tazlin)
- fix: stability fixes [`#150`](https://github.com/Haidra-Org/hordelib/pull/150) (Jug)
- fix: Tox lint/style environments now build (more) correctly [`#151`](https://github.com/Haidra-Org/hordelib/pull/151) (tazlin)
- fix: don't mix up controlnets and run out of vram [`#147`](https://github.com/Haidra-Org/hordelib/pull/147) (Jug)
- feat: active memory and model management [`#144`](https://github.com/Haidra-Org/hordelib/pull/144) (Jug)
- fix: Make thread locking as minimalist as possible [`#142`](https://github.com/Haidra-Org/hordelib/pull/142) (Jug)
- fix: Sha validation fix [`#139`](https://github.com/Haidra-Org/hordelib/pull/139) (tazlin)
- fix: threading and job settings being mixed together [`#127`](https://github.com/Haidra-Org/hordelib/pull/127) (Jug)
- feat: performance optimisation [`#125`](https://github.com/Haidra-Org/hordelib/pull/125) (Jug)
- refactor: Logger tweaks, Model Manager housekeeping [`#118`](https://github.com/Haidra-Org/hordelib/pull/118) (tazlin)
- feat: Clip Rankings [`#117`](https://github.com/Haidra-Org/hordelib/pull/117) (Divided by Zer0)
- feat: Blip [`#116`](https://github.com/Haidra-Org/hordelib/pull/116) (Divided by Zer0)
- build: add support for production builld [`#109`](https://github.com/Haidra-Org/hordelib/pull/109) (Jug)
- feat: make logging setup and control optional [`#106`](https://github.com/Haidra-Org/hordelib/pull/106) (Jug)
- fix: suppress terminal spam [`#104`](https://github.com/Haidra-Org/hordelib/pull/104) (Jug)
- feat: add support for separate source_mask [`#103`](https://github.com/Haidra-Org/hordelib/pull/103) (Jug)
- fix: img2img + highres_fix  [`#80`](https://github.com/Haidra-Org/hordelib/pull/80) (Divided by Zer0)
- tests: class scope on inference tests for speedup [`#78`](https://github.com/Haidra-Org/hordelib/pull/78) (Divided by Zer0)
- build: add release mode flag [`#76`](https://github.com/Haidra-Org/hordelib/pull/76) (Jug)
- refactor!: Second big Model Manager rework step [`#75`](https://github.com/Haidra-Org/hordelib/pull/75) (tazlin)
- fix: adjust mlsd annotator defaults [`#74`](https://github.com/Haidra-Org/hordelib/pull/74) (Jug)
- build: patch release [`#73`](https://github.com/Haidra-Org/hordelib/pull/73) (Jug)
- build: upgrade to torch 2, xformers 18 and latest comfyui [`#68`](https://github.com/Haidra-Org/hordelib/pull/68) (Jug)
- feat: Added is_model_loaded() to HyperMM [`#67`](https://github.com/Haidra-Org/hordelib/pull/67) (Divided by Zer0)
- feat: add support for return_control_map [`#66`](https://github.com/Haidra-Org/hordelib/pull/66) (Jug)
- fix: resize img2img before inference [`#63`](https://github.com/Haidra-Org/hordelib/pull/63) (Divided by Zer0)
- fix: add timezone to build results [`#61`](https://github.com/Haidra-Org/hordelib/pull/61) (Jug)
- tests: gfpgan test and size assets [`#62`](https://github.com/Haidra-Org/hordelib/pull/62) (Divided by Zer0)
- feat: Make use of the ControlNet ModelManager [`#53`](https://github.com/Haidra-Org/hordelib/pull/53) (Divided by Zer0)
- test: fix test with red border around it [`#58`](https://github.com/Haidra-Org/hordelib/pull/58) (Jug)
- build: activate build results website [`#57`](https://github.com/Haidra-Org/hordelib/pull/57) (Jug)
- build: make a webpage of test result images [`#55`](https://github.com/Haidra-Org/hordelib/pull/55) (Jug)
- test: fix black 64x64 image tests [`#54`](https://github.com/Haidra-Org/hordelib/pull/54) (Jug)
- feat: add face fixing support [`#50`](https://github.com/Haidra-Org/hordelib/pull/50) (Jug)
- test: change all tests to webp [`#49`](https://github.com/Haidra-Org/hordelib/pull/49) (Jug)
- feat: add controlnet support [`#46`](https://github.com/Haidra-Org/hordelib/pull/46) (Jug)
- ci: inpainting tests [`#47`](https://github.com/Haidra-Org/hordelib/pull/47) (Divided by Zer0)
- build: change how custom nodes are loaded into comfyui [`#44`](https://github.com/Haidra-Org/hordelib/pull/44) (Jug)
- ci: Disable pypi publish [`#45`](https://github.com/Haidra-Org/hordelib/pull/45) (Divided by Zer0)
- docs: readme updates. [`#43`](https://github.com/Haidra-Org/hordelib/pull/43) (Jug)
- docs: readme updates. [`#42`](https://github.com/Haidra-Org/hordelib/pull/42) (Jug)
- feat: Re-adds diffusers model manager [`#41`](https://github.com/Haidra-Org/hordelib/pull/41) (tazlin)
- test: add diffusers inpainting run example [`#40`](https://github.com/Haidra-Org/hordelib/pull/40) (Jug)
- docs: update readme [`#39`](https://github.com/Haidra-Org/hordelib/pull/39) (Jug)
- refactor: We do some light refactoring... [`#34`](https://github.com/Haidra-Org/hordelib/pull/34) (Divided by Zer0)
- test: Optimized tests [`#32`](https://github.com/Haidra-Org/hordelib/pull/32) (Divided by Zer0)
- refactor: Significant code cleanup and CI/build improvements. [`#30`](https://github.com/Haidra-Org/hordelib/pull/30) (tazlin)
- feat: Post processors [`#27`](https://github.com/Haidra-Org/hordelib/pull/27) (Divided by Zer0)
- fix: Duplicate lines [`#25`](https://github.com/Haidra-Org/hordelib/pull/25) (tazlin)
- feat: Adds a github action when pushing to main that will generate a new release and an automatic changelog [`#24`](https://github.com/Haidra-Org/hordelib/pull/24) (Jug)
- fix: References to `horde_model_manager`, more docs [`#23`](https://github.com/Haidra-Org/hordelib/pull/23) (tazlin)
- docs: Update LICENSE [`#20`](https://github.com/Haidra-Org/hordelib/pull/20) (tazlin)
- refactor: ModelManager improvements, test adjustments [`#19`](https://github.com/Haidra-Org/hordelib/pull/19) (tazlin)
- fix: missing return [`#18`](https://github.com/Haidra-Org/hordelib/pull/18) (Divided by Zer0)
- refactor: 'ModelManager' rework, added 'WorkerSettings' [`#17`](https://github.com/Haidra-Org/hordelib/pull/17) (tazlin)
- refactor: Test tweaks, type hint fixes [`#16`](https://github.com/Haidra-Org/hordelib/pull/16) (tazlin)
- refactor: Type hints, refactoring, preemptive checks [`#15`](https://github.com/Haidra-Org/hordelib/pull/15) (tazlin)
- fix: test_horde.py syntax error [`#14`](https://github.com/Haidra-Org/hordelib/pull/14) (tazlin)
- fix: Compat fixes for tests from pr #11 [`#12`](https://github.com/Haidra-Org/hordelib/pull/12) (tazlin)
- feat: Clip interrogation support [`#11`](https://github.com/Haidra-Org/hordelib/pull/11) (tazlin)
- feat: Adds support for using a Model Manager  [`#8`](https://github.com/Haidra-Org/hordelib/pull/8) (Divided by Zer0)
- fix: add proper exception logging to comfyui, closes #64 [`#64`](https://github.com/Haidra-Org/hordelib/issues/64)  ()
- fix: untrack automatically downloaded model reference jsons [`501f35d`](https://github.com/Haidra-Org/hordelib/commit/501f35d897153bb95ef50c32be8a3045449fb4c1)  (tazlin)
- fix: remove unused model 'db.json' [`6a5b29d`](https://github.com/Haidra-Org/hordelib/commit/6a5b29dc005e30f854b09070348c30fcbb7a5638)  (tazlin)
- Revert "fix: pin timm version to 0.6.13" [`9c82655`](https://github.com/Haidra-Org/hordelib/commit/9c82655ac5f160d8676ade4611c7f157cdde5875)  (Jug)

Generated by [`auto-changelog`](https://github.com/CookPete/auto-changelog).
