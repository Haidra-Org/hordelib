## hordelib Changelog

## [v0.10.0](https://github.com/jug-dev/hordelib/compare/v0.9.5...v0.10.0)

22 April 2023

- feat: add dynamic prompt support [`#161`](https://github.com/jug-dev/hordelib/pull/161) (Jug)
- fix: stability fixes [`#159`](https://github.com/jug-dev/hordelib/pull/159) (Jug)
- fix: Moves ControlNet Annotators to `AIWORKER_CACHE_HOME` [`e0a7db7`](https://github.com/jug-dev/hordelib/commit/e0a7db7e89dc91016dd5abba2e94ccf3767c65d3)  (tazlin)
- refactor: cleans up the preload annotators functions [`6b8f9c4`](https://github.com/jug-dev/hordelib/commit/6b8f9c453ef38af3439a6bf6ca2b99de649fc6c6)  (tazlin)
- feat: Preload controlnet annotators [`6d16515`](https://github.com/jug-dev/hordelib/commit/6d1651535915022cac8e7d218ca4d5081d1d2241)  (tazlin)

## [v0.9.5](https://github.com/jug-dev/hordelib/compare/v0.9.4...v0.9.5)

20 April 2023

- build: fix missing dependency in pypi build [`01169be`](https://github.com/jug-dev/hordelib/commit/01169beb575c0a0577eebc00210ca98d14eeff42)  (Jug)

## [v0.9.4](https://github.com/jug-dev/hordelib/compare/v0.9.3...v0.9.4)

20 April 2023

- fix: add missing dependency [`1dc5127`](https://github.com/jug-dev/hordelib/commit/1dc51270a15981aa6eb08aafe2ae8fba7a1e7d57)  (Jug)

## [v0.9.3](https://github.com/jug-dev/hordelib/compare/v0.9.2...v0.9.3)

20 April 2023

- CI: trigger CI with certain other critical files [`#152`](https://github.com/jug-dev/hordelib/pull/152) (tazlin)
- fix: stability fixes [`#150`](https://github.com/jug-dev/hordelib/pull/150) (Jug)
- fix: Tox lint/style environments now build (more) correctly [`#151`](https://github.com/jug-dev/hordelib/pull/151) (tazlin)
- Revert "Merge branch 'releases' into main" [`27987a0`](https://github.com/jug-dev/hordelib/commit/27987a0884b0ce48cad9d458c9fbdfa9b423b4a2)  (Jug)
- refactor: Housekeeping, preparing for full lint ruleset in CI [`06155d2`](https://github.com/jug-dev/hordelib/commit/06155d26eb0b81ba704466adfefe34fd7347e42f)  (tazlin)
- refactor: Control net model manager housekeeping [`37b6cac`](https://github.com/jug-dev/hordelib/commit/37b6cac2c1ebf79479996aecec1c4414d9c8b243)  (tazlin)

## [v0.9.2](https://github.com/jug-dev/hordelib/compare/v0.9.1...v0.9.2)

17 April 2023

- fix: don't mix up controlnets and run out of vram [`#147`](https://github.com/jug-dev/hordelib/pull/147) (Jug)

## [v0.9.1](https://github.com/jug-dev/hordelib/compare/v0.9.0...v0.9.1)

17 April 2023

- fix: add proper exception logging to comfyui, closes #64 [`#64`](https://github.com/jug-dev/hordelib/issues/64)  ()

## [v0.9.0](https://github.com/jug-dev/hordelib/compare/v0.8.8...v0.9.0)

16 April 2023

- feat: active memory and model management [`#144`](https://github.com/jug-dev/hordelib/pull/144) (Jug)

## [v0.8.8](https://github.com/jug-dev/hordelib/compare/v0.8.7...v0.8.8)

15 April 2023

- fix: Make thread locking as minimalist as possible [`#142`](https://github.com/jug-dev/hordelib/pull/142) (Jug)
- fix: fix broken stress test [`be9567e`](https://github.com/jug-dev/hordelib/commit/be9567e8935f6016a607ad7de58522d2d84b21f7)  (Jug)

## [v0.8.7](https://github.com/jug-dev/hordelib/compare/v0.8.6...v0.8.7)

15 April 2023

- fix: don't thread lock loading with inference [`00ec98d`](https://github.com/jug-dev/hordelib/commit/00ec98dc8c80c14208f38c58bc7673ad03d7ab4b)  (Jug)
- chore: more badge refresh tweaks [`4a54868`](https://github.com/jug-dev/hordelib/commit/4a54868cb05b5ac8c45e6646ff04565c28732df2)  (Jug)

## [v0.8.6](https://github.com/jug-dev/hordelib/compare/v0.8.5...v0.8.6)

15 April 2023

- fix: Sha validation fix [`#139`](https://github.com/jug-dev/hordelib/pull/139) (tazlin)
- fix: pytest discovery, broken by non-tests in test folder [`534695e`](https://github.com/jug-dev/hordelib/commit/534695e10e5f6a94571059a6916d9798867860fa)  (tazlin)
- fix: switches pr CI to use example/ run_* [`e6ba23e`](https://github.com/jug-dev/hordelib/commit/e6ba23e62bc44eaebe0996201baf807a150a389e)  (tazlin)
- build: update CI to do a weak lint/formatting check [`6d09423`](https://github.com/jug-dev/hordelib/commit/6d09423e3c8028bc17d0be36f460b9a1963207f1)  (tazlin)

## [v0.8.5](https://github.com/jug-dev/hordelib/compare/v0.8.4...v0.8.5)

14 April 2023

- test: add threaded torture test [`03bd56a`](https://github.com/jug-dev/hordelib/commit/03bd56a3c40d8ace9dbbb94c0747e207c8442e4a)  (Jug)
- fix: assert parameter bounds to stop errors [`d1a18f7`](https://github.com/jug-dev/hordelib/commit/d1a18f7659336614738d9106f7ffccc72c1b9de0)  (Jug)

## [v0.8.4](https://github.com/jug-dev/hordelib/compare/v0.8.3...v0.8.4)

14 April 2023

- fix: threading and job settings being mixed together [`#127`](https://github.com/jug-dev/hordelib/pull/127) (Jug)
- ci: try to refresh pypi badge on release [`fbb832c`](https://github.com/jug-dev/hordelib/commit/fbb832c8cf3906958ed08b6fb0408811130e57ca)  (Jug)
- docs: minor url tweak in readme [`6faee9a`](https://github.com/jug-dev/hordelib/commit/6faee9a8757b749013f3809c2617e7a7003310ba)  (Jug)

## [v0.8.3](https://github.com/jug-dev/hordelib/compare/v0.8.2...v0.8.3)

13 April 2023

- fix: defer model manager loading [`5c41949`](https://github.com/jug-dev/hordelib/commit/5c419490e708fa7fe0778d97b1c53cc5ec7aa63e)  (tazlin)

## [v0.8.2](https://github.com/jug-dev/hordelib/compare/v0.8.1...v0.8.2)

13 April 2023

- feat: performance optimisation [`#125`](https://github.com/jug-dev/hordelib/pull/125) (Jug)
- refactor: Logger tweaks, Model Manager housekeeping [`#118`](https://github.com/jug-dev/hordelib/pull/118) (tazlin)
- docs: Update README.md [`1fea1f1`](https://github.com/jug-dev/hordelib/commit/1fea1f1abd9372fb56711b67145f228c5987db07)  (tazlin)
- ci: remove junk from changelog [`f943b8f`](https://github.com/jug-dev/hordelib/commit/f943b8f5d3169f37c58f6415f2bb39314706dc12)  (Jug)

## [v0.8.1](https://github.com/jug-dev/hordelib/compare/v0.8.0...v0.8.1)

13 April 2023

- fix: suppress terminal spam from comfyui [`07764b5`](https://github.com/jug-dev/hordelib/commit/07764b51eb2a78c1557d5acceec265b3a4cd47da)  (Jug)

## [v0.8.0](https://github.com/jug-dev/hordelib/compare/v0.7.3...v0.8.0)

13 April 2023

- feat: Clip Rankings [`#117`](https://github.com/jug-dev/hordelib/pull/117) (Divided by Zer0)
- feat: Blip [`#116`](https://github.com/jug-dev/hordelib/pull/116) (Divided by Zer0)
- fix: make library thread safe [`b7274b4`](https://github.com/jug-dev/hordelib/commit/b7274b4dc1170480513fbd7eb34a4e6a0cd3aaf4)  (Jug)
- fix: remove thread mutex for now [`1a901b8`](https://github.com/jug-dev/hordelib/commit/1a901b8d8274f84663dd2933224d375105120a1b)  (Jug)
- build: fix build_helper for local use [`6ec5f38`](https://github.com/jug-dev/hordelib/commit/6ec5f38279eca2643fbfdf0ff9cc1ab7031b6bb3)  (Jug)

## [v0.7.3](https://github.com/jug-dev/hordelib/compare/v0.7.2...v0.7.3)

12 April 2023

- build: more production build fixes. [`0ca0367`](https://github.com/jug-dev/hordelib/commit/0ca0367e98ec88f9b30428f797101e333902a0f9)  (Jug)

## [v0.7.2](https://github.com/jug-dev/hordelib/compare/v0.7.1...v0.7.2)

12 April 2023

- build: fix production build packaging [`1236763`](https://github.com/jug-dev/hordelib/commit/12367632c4248562423c8745fa3534bc19d0c31e)  (Jug)

## [v0.7.1](https://github.com/jug-dev/hordelib/compare/v0.7.0...v0.7.1)

12 April 2023

- build: fix missing build time dependency [`de649d4`](https://github.com/jug-dev/hordelib/commit/de649d426c79140a85e3f70e60928628061422d5)  (Jug)

## [v0.7.0](https://github.com/jug-dev/hordelib/compare/v0.6.1...v0.7.0)

12 April 2023

- build: add support for production builld [`#109`](https://github.com/jug-dev/hordelib/pull/109) (Jug)
- build: fix detection of production build [`029bcac`](https://github.com/jug-dev/hordelib/commit/029bcac3b1dd86a06fa453db4e804818929d67bd)  (Jug)
- ci: if a label is forgotten on release, assume patch release [`57bdba3`](https://github.com/jug-dev/hordelib/commit/57bdba3f99c783bde769e5857762033d64b1acf1)  (Jug)

## [v0.6.1](https://github.com/jug-dev/hordelib/compare/v0.6.0...v0.6.1)

12 April 2023

- feat: make logging setup and control optional [`#106`](https://github.com/jug-dev/hordelib/pull/106) (Jug)
- style: Automatic formatting/lint with length 119 [`845764a`](https://github.com/jug-dev/hordelib/commit/845764ad5765f2b4e7c442a25f421af493fc7c88)  (tazlin)
- docs: cleanup readme for viewing in an editor [`76a7f7f`](https://github.com/jug-dev/hordelib/commit/76a7f7f7198da1f988d1972e2b9082944b97af3a)  (Jug)
- chore: Change black to line length 119 [`429aadf`](https://github.com/jug-dev/hordelib/commit/429aadff04e48948167173497d134983a9dc26c4)  (tazlin)

## [v0.6.0](https://github.com/jug-dev/hordelib/compare/v0.5.24...v0.6.0)

12 April 2023

- fix: suppress terminal spam [`#104`](https://github.com/jug-dev/hordelib/pull/104) (Jug)
- feat: add support for separate source_mask [`#103`](https://github.com/jug-dev/hordelib/pull/103) (Jug)

## [v0.5.24](https://github.com/jug-dev/hordelib/compare/v0.5.23...v0.5.24)

12 April 2023

- ci: include changelog link on pypi page [`a617c23`](https://github.com/jug-dev/hordelib/commit/a617c2381a46b9e3b421074dfb5162c3a600c131)  (Jug)

## [v0.5.23](https://github.com/jug-dev/hordelib/compare/v0.5.22...v0.5.23)

12 April 2023

- ci: customise the changelog format [`c85285a`](https://github.com/jug-dev/hordelib/commit/c85285ab213862d32dde591cbf714e2d2c3dd3ba)  (Jug)

## [v0.5.22](https://github.com/jug-dev/hordelib/compare/v0.5.21...v0.5.22)

12 April 2023

## [v0.5.21](https://github.com/jug-dev/hordelib/compare/v0.5.20...v0.5.21)

12 April 2023

- ci: try to generate changelog for the right version [`2b86531`](https://github.com/jug-dev/hordelib/commit/2b865314cb8f6f387e181a87771739cf983f1fa6)  (Jug)

## [v0.5.20](https://github.com/jug-dev/hordelib/compare/v0.5.19...v0.5.20)

12 April 2023

- ci: try a better changelog generator [`a03b5fc`](https://github.com/jug-dev/hordelib/commit/a03b5fcd052269451004e4c9f45ef5775f5331d8)  (Jug)
- ci: more tweaks [`81ac9b5`](https://github.com/jug-dev/hordelib/commit/81ac9b587fdeacf1159c9a01f7ef9c06a24bcc52)  (Jug)
- ci: ci again [`a185c41`](https://github.com/jug-dev/hordelib/commit/a185c410e3cadcd56bd5b2d63712abd2859884e5)  (Jug)

## [v0.5.19](https://github.com/jug-dev/hordelib/compare/v0.5.18...v0.5.19)

12 April 2023

- ci: changelog wasn't include in setuptools [`652e53c`](https://github.com/jug-dev/hordelib/commit/652e53cf4db43e6e5502edc05b299da3e8f63644)  (Jug)

## [v0.5.18](https://github.com/jug-dev/hordelib/compare/v0.5.17...v0.5.18)

11 April 2023

- ci: release ci tweaks [`1adf7ce`](https://github.com/jug-dev/hordelib/commit/1adf7ce5d1c607ea04011803324e20b26e348a3c)  (Jug)

## [v0.5.17](https://github.com/jug-dev/hordelib/compare/v0.5.16...v0.5.17)

11 April 2023

- style: Incremental style/lint catchup [`1cb70d2`](https://github.com/jug-dev/hordelib/commit/1cb70d2eb3c219bff304c4ed0ac34f7456946281)  (tazlin)
- ci: Enables a couple ruff rules for CI [`17bd0f8`](https://github.com/jug-dev/hordelib/commit/17bd0f8b1cfd73f3e1429ad4c9b007623517de35)  (tazlin)
- ci: another day another way to do changelogs [`6e7ed60`](https://github.com/jug-dev/hordelib/commit/6e7ed604fee66388570b9e953f25599d49a3bc9a)  (Jug)

## [v0.5.16](https://github.com/jug-dev/hordelib/compare/v0.5.15...v0.5.16)

11 April 2023

- ci: more ci tweaks [`7bbc0c3`](https://github.com/jug-dev/hordelib/commit/7bbc0c3491c12e00ee54fa5f96e73fcf801ae6f7)  (Jug)

## [v0.5.15](https://github.com/jug-dev/hordelib/compare/v0.5.14...v0.5.15)

11 April 2023

- ci: this is never going to work is it [`797a317`](https://github.com/jug-dev/hordelib/commit/797a3172666e4eab9fe88bc22307e7ee84063441)  (Jug)

## [v0.5.14](https://github.com/jug-dev/hordelib/compare/v0.5.13...v0.5.14)

11 April 2023

- ci: another day another ci hack [`bf007ac`](https://github.com/jug-dev/hordelib/commit/bf007ace0740eeebf17dd95013d5d68cf332a209)  (Jug)

## [v0.5.13](https://github.com/jug-dev/hordelib/compare/v0.5.12...v0.5.13)

11 April 2023

- ci: optimistically try to output a changelog [`cfc71a2`](https://github.com/jug-dev/hordelib/commit/cfc71a28468b391e70bd85a8b3b57550f51ec328)  (Jug)

## [v0.5.12](https://github.com/jug-dev/hordelib/compare/v0.5.11...v0.5.12)

11 April 2023

- ci: Try harder to generate a changelog [`4eff72d`](https://github.com/jug-dev/hordelib/commit/4eff72df9c5ea8c811a4fb09099aa2bc10319fc5)  (Jug)

## [v0.5.11](https://github.com/jug-dev/hordelib/compare/v0.5.10...v0.5.11)

11 April 2023

- ci: add some notes to the release ci [`faf9788`](https://github.com/jug-dev/hordelib/commit/faf9788de1e284a1d0b4b54b5c36b1a5fc789ff9)  (Jug)
- ci: generate a changelog once again [`8749370`](https://github.com/jug-dev/hordelib/commit/8749370ca78a0e5413293f5ca1a9a60859fd5e97)  (Jug)

## [v0.5.10](https://github.com/jug-dev/hordelib/compare/v0.5.9...v0.5.10)

11 April 2023

- docs: remove changelog [`62dae03`](https://github.com/jug-dev/hordelib/commit/62dae03cd056ee19a7a433a1c360e026266329a7)  (Jug)
- ci: tweak release scripts [`bf9a6a7`](https://github.com/jug-dev/hordelib/commit/bf9a6a76a7014e263a3554fefe5c8195c782f6f0)  (Jug)

## [v0.5.9](https://github.com/jug-dev/hordelib/compare/v0.5.8...v0.5.9)

11 April 2023

- ci: more tweaks to the ci process [`522d269`](https://github.com/jug-dev/hordelib/commit/522d2699385c32b89cbcdf45d89bb6801daaadfb)  (Jug)

## [v0.5.8](https://github.com/jug-dev/hordelib/compare/v0.5.7...v0.5.8)

11 April 2023

## [v0.5.7](https://github.com/jug-dev/hordelib/compare/v0.5.6...v0.5.7)

11 April 2023

- fix: img2img + highres_fix  [`#80`](https://github.com/jug-dev/hordelib/pull/80) (Divided by Zer0)
- ci: try to publish to pypi on release [`deb6eb5`](https://github.com/jug-dev/hordelib/commit/deb6eb5f7661be5ebbb91e121c382338a89ecb76)  (Jug)
- ci: tweaks to the release ci [`c234ea7`](https://github.com/jug-dev/hordelib/commit/c234ea7157b4c70e62adbc2b71d7220a86e1ec98)  (Jug)

## [v0.5.6](https://github.com/jug-dev/hordelib/compare/v0.5.5...v0.5.6)

11 April 2023

- tests: class scope on inference tests for speedup [`#78`](https://github.com/jug-dev/hordelib/pull/78) (Divided by Zer0)
- docs: recreate LICENSE [`158a70f`](https://github.com/jug-dev/hordelib/commit/158a70f32ab27dbea6eae9d37c2eddad90016263)  (Jug)
- docs: remove license to recreate it [`8ee4dde`](https://github.com/jug-dev/hordelib/commit/8ee4ddeb40b1824d029a7eca138cd76e61b484f0)  (Jug)
- build: placeholder changelog [`bbf880e`](https://github.com/jug-dev/hordelib/commit/bbf880e011ea34325beb0bbb46a0ad1545e25af0)  (Jug)

## [v0.5.5](https://github.com/jug-dev/hordelib/compare/v0.5.4...v0.5.5)

11 April 2023

## [v0.5.4](https://github.com/jug-dev/hordelib/compare/v0.5.3...v0.5.4)

11 April 2023

- build: add release mode flag [`#76`](https://github.com/jug-dev/hordelib/pull/76) (Jug)
- refactor!: Second big Model Manager rework step [`#75`](https://github.com/jug-dev/hordelib/pull/75) (tazlin)
- fix: adjust mlsd annotator defaults [`#74`](https://github.com/jug-dev/hordelib/pull/74) (Jug)
- chore: resolve merge conflicts [`007bc44`](https://github.com/jug-dev/hordelib/commit/007bc441ff448eda879994212fc0e5ad896b4e84)  (Jug)
- docs: remove the changelog from main [`6a650f2`](https://github.com/jug-dev/hordelib/commit/6a650f215482c316f77de2e8fb98609c4d60fbc7)  (Jug)
- fix: normal map and mlsd annotators [`203873d`](https://github.com/jug-dev/hordelib/commit/203873dc6e904f73f48c56d1b9f68509e3213c15)  (Jug)

## [v0.5.3](https://github.com/jug-dev/hordelib/compare/v0.5.2...v0.5.3)

11 April 2023

- build: patch release [`#73`](https://github.com/jug-dev/hordelib/pull/73) (Jug)
- build: try to fix test running and build [`9df056d`](https://github.com/jug-dev/hordelib/commit/9df056d98d4502398bb13437f51522cb6a0feebf)  (Jug)

## [v0.5.2](https://github.com/jug-dev/hordelib/compare/v0.5.1...v0.5.2)

11 April 2023

- build: upgrade to torch 2, xformers 18 and latest comfyui [`#68`](https://github.com/jug-dev/hordelib/pull/68) (Jug)

## [v0.5.1](https://github.com/jug-dev/hordelib/compare/v0.5.0...v0.5.1)

11 April 2023

- feat: Added is_model_loaded() to HyperMM [`#67`](https://github.com/jug-dev/hordelib/pull/67) (Divided by Zer0)

## [v0.5.0](https://github.com/jug-dev/hordelib/compare/v0.4.2...v0.5.0)

11 April 2023

- feat: add support for return_control_map [`#66`](https://github.com/jug-dev/hordelib/pull/66) (Jug)
- docs: update ci test badge [`e2b137e`](https://github.com/jug-dev/hordelib/commit/e2b137ed43d5b63c8a6f1d7899a31bbb78aa7045)  (Jug)

## [v0.4.2](https://github.com/jug-dev/hordelib/compare/v0.4.1...v0.4.2)

11 April 2023

- fix: resize img2img before inference [`#63`](https://github.com/jug-dev/hordelib/pull/63) (Divided by Zer0)
- fix: add timezone to build results [`#61`](https://github.com/jug-dev/hordelib/pull/61) (Jug)
- tests: gfpgan test and size assets [`#62`](https://github.com/jug-dev/hordelib/pull/62) (Divided by Zer0)
- docs: update with pypi test notes [`dd41120`](https://github.com/jug-dev/hordelib/commit/dd4112023d39e280fb61a1342707af4765f3b4df)  (Jug)

## [v0.4.1](https://github.com/jug-dev/hordelib/compare/v0.4.0...v0.4.1)

10 April 2023

- feat: Make use of the ControlNet ModelManager [`#53`](https://github.com/jug-dev/hordelib/pull/53) (Divided by Zer0)
- test: fix test with red border around it [`#58`](https://github.com/jug-dev/hordelib/pull/58) (Jug)
- build: activate build results website [`#57`](https://github.com/jug-dev/hordelib/pull/57) (Jug)
- build: make a webpage of test result images [`#55`](https://github.com/jug-dev/hordelib/pull/55) (Jug)
- test: fix black 64x64 image tests [`#54`](https://github.com/jug-dev/hordelib/pull/54) (Jug)
- version incremented [`14efa65`](https://github.com/jug-dev/hordelib/commit/14efa65031d595179cd00b4a1f27b9bce6ab88ac)  (github-actions)
- build: try to be smarter when we run tests [`e0d9d4b`](https://github.com/jug-dev/hordelib/commit/e0d9d4bea09233f0f59ac28a413dd754eab613b8)  (Jug)
- build: try to run tests more often [`4470a24`](https://github.com/jug-dev/hordelib/commit/4470a243135c47d1bc24859573c6676e58dfc64c)  (Jug)

## [v0.4.0](https://github.com/jug-dev/hordelib/compare/v0.3.1...v0.4.0)

10 April 2023

- feat: add face fixing support [`#50`](https://github.com/jug-dev/hordelib/pull/50) (Jug)
- version incremented [`c5aa1d5`](https://github.com/jug-dev/hordelib/commit/c5aa1d5614738cc50418772dcf70343e614e2e9d)  (github-actions)

## [v0.3.1](https://github.com/jug-dev/hordelib/compare/v0.3.0...v0.3.1)

10 April 2023

- test: change all tests to webp [`#49`](https://github.com/jug-dev/hordelib/pull/49) (Jug)
- version incremented [`a1adb39`](https://github.com/jug-dev/hordelib/commit/a1adb398cdc55f242158d6d3e2f6ec11645af892)  (github-actions)

## [v0.3.0](https://github.com/jug-dev/hordelib/compare/v0.2.2...v0.3.0)

10 April 2023

- feat: add controlnet support [`#46`](https://github.com/jug-dev/hordelib/pull/46) (Jug)
- version incremented [`62f72c1`](https://github.com/jug-dev/hordelib/commit/62f72c10373711df96d43f89c1834f2d4dee3cf5)  (github-actions)
- docs: add build status badge to readme [`86b3d1a`](https://github.com/jug-dev/hordelib/commit/86b3d1a61442733b0a81f97f02ca58f8196f2f1c)  (Jug)

## [v0.2.2](https://github.com/jug-dev/hordelib/compare/v0.2.1...v0.2.2)

9 April 2023

- ci: inpainting tests [`#47`](https://github.com/jug-dev/hordelib/pull/47) (Divided by Zer0)
- version incremented [`5940da8`](https://github.com/jug-dev/hordelib/commit/5940da8efeae955a45dc84ea51905d700df5b190)  (github-actions)

## [v0.2.1](https://github.com/jug-dev/hordelib/compare/v0.2.0...v0.2.1)

9 April 2023

- build: change how custom nodes are loaded into comfyui [`#44`](https://github.com/jug-dev/hordelib/pull/44) (Jug)

## [v0.2.0](https://github.com/jug-dev/hordelib/compare/v0.1.0...v0.2.0)

9 April 2023

- ci: Disable pypi publish [`#45`](https://github.com/jug-dev/hordelib/pull/45) (Divided by Zer0)
- docs: readme updates. [`#43`](https://github.com/jug-dev/hordelib/pull/43) (Jug)
- docs: readme updates. [`#42`](https://github.com/jug-dev/hordelib/pull/42) (Jug)
- feat: Re-adds diffusers model manager [`#41`](https://github.com/jug-dev/hordelib/pull/41) (tazlin)
- test: add diffusers inpainting run example [`#40`](https://github.com/jug-dev/hordelib/pull/40) (Jug)
- docs: update readme [`#39`](https://github.com/jug-dev/hordelib/pull/39) (Jug)
- refactor: We do some light refactoring... [`#34`](https://github.com/jug-dev/hordelib/pull/34) (Divided by Zer0)
- test: Optimized tests [`#32`](https://github.com/jug-dev/hordelib/pull/32) (Divided by Zer0)
- refactor: Significant code cleanup and CI/build improvements. [`#30`](https://github.com/jug-dev/hordelib/pull/30) (tazlin)
- feat: Post processors [`#27`](https://github.com/jug-dev/hordelib/pull/27) (Divided by Zer0)
- feat: adds inpainting [`dea0e9e`](https://github.com/jug-dev/hordelib/commit/dea0e9e54c8fcca87f4bb385fd68b811e8eb9a4e)  (Jug)
- feat: image loader, basic img2img [`04994ea`](https://github.com/jug-dev/hordelib/commit/04994eaa4cd224071630f479f49d3ca578cb920a)  (Jug)
- test: reduce vram requirements for hires fix tests [`2932764`](https://github.com/jug-dev/hordelib/commit/2932764a6e7bc5af271f883a11cb73ee94b5fb12)  (Jug)

## [v0.1.0](https://github.com/jug-dev/hordelib/compare/v0.0.10...v0.1.0)

6 April 2023

- fix: Duplicate lines [`#25`](https://github.com/jug-dev/hordelib/pull/25) (tazlin)
- feat: Adds a github action when pushing to main that will generate a new release and an automatic changelog [`#24`](https://github.com/jug-dev/hordelib/pull/24) (Jug)
- fix: References to `horde_model_manager`, more docs [`#23`](https://github.com/jug-dev/hordelib/pull/23) (tazlin)
- docs: Update LICENSE [`#20`](https://github.com/jug-dev/hordelib/pull/20) (tazlin)
- refactor: ModelManager improvements, test adjustments [`#19`](https://github.com/jug-dev/hordelib/pull/19) (tazlin)
- fix: missing return [`#18`](https://github.com/jug-dev/hordelib/pull/18) (Divided by Zer0)
- refactor: 'ModelManager' rework, added 'WorkerSettings' [`#17`](https://github.com/jug-dev/hordelib/pull/17) (tazlin)
- refactor: Test tweaks, type hint fixes [`#16`](https://github.com/jug-dev/hordelib/pull/16) (tazlin)
- refactor: Type hints, refactoring, preemptive checks [`#15`](https://github.com/jug-dev/hordelib/pull/15) (tazlin)
- feat: adds clip skip support [`dd1cbcc`](https://github.com/jug-dev/hordelib/commit/dd1cbcc44b9c8558f3961de1c99e25b548170066)  (Jug)
- refactor: make things more explicit. [`970fd4a`](https://github.com/jug-dev/hordelib/commit/970fd4a21b6b771fe5e83769a74a6ae4b9be2aec)  (Jug)
- feat: allow running comfyui easily. [`3ce7af1`](https://github.com/jug-dev/hordelib/commit/3ce7af146462bf2140b85514fef21314f2b8bdaa)  (Jug)

## [v0.0.10](https://github.com/jug-dev/hordelib/compare/v0.0.9...v0.0.10)

3 April 2023

- fix: test_horde.py syntax error [`#14`](https://github.com/jug-dev/hordelib/pull/14) (tazlin)
- fix: Compat fixes for tests from pr #11 [`#12`](https://github.com/jug-dev/hordelib/pull/12) (tazlin)
- feat: Clip interrogation support [`#11`](https://github.com/jug-dev/hordelib/pull/11) (tazlin)
- feat: Adds support for using a Model Manager  [`#8`](https://github.com/jug-dev/hordelib/pull/8) (Divided by Zer0)
- build: fixes for new model manager and clip [`9d09885`](https://github.com/jug-dev/hordelib/commit/9d0988505a845a433b60c9f9fe1d8e9784c8ced9)  (Jug)
- build: update comfyui to latest version [`a5cfc05`](https://github.com/jug-dev/hordelib/commit/a5cfc05352f8a86c9beb511ab5b869a2c72b6cb3)  (Jug)
- build: disable forced reformatting from black [`835ffe5`](https://github.com/jug-dev/hordelib/commit/835ffe546ec7dd43342bc133aeeaf6b625b8e357)  (Jug)

## [v0.0.9](https://github.com/jug-dev/hordelib/compare/v0.0.8...v0.0.9)

3 April 2023

- test: More thorough tests for ComfyUI install [`a247f2b`](https://github.com/jug-dev/hordelib/commit/a247f2b9fd9b62c3b0718468a50859e06c92ee65)  (Jug)
- build: adds build helper script. [`afd38ea`](https://github.com/jug-dev/hordelib/commit/afd38eabbf103b52c32a8e4daff2e7f1e7b2324d)  (Jug)
- build: don't run inference tests on github (no cuda) [`638056b`](https://github.com/jug-dev/hordelib/commit/638056bc2d48098a4cdcad592ab955abe815fcd7)  (Jug)

## v0.0.8

2 April 2023

- Initial commit. [`e3eae1c`](https://github.com/jug-dev/hordelib/commit/e3eae1c452e0f3145af3b8b62c14c377b3136f7b)  (Jug)
- feat: Support loading ComfyUI pipelines without manual modification. [`8c2fd13`](https://github.com/jug-dev/hordelib/commit/8c2fd13db4ce6a293f335cefaaf9c7d52aaefd7b)  (Jug)
- feat: adds stable difussion hires fix pipeline. [`6e573cb`](https://github.com/jug-dev/hordelib/commit/6e573cb364343569ad59cbd709b266d292021428)  (Jug)

Generated by [`auto-changelog`](https://github.com/CookPete/auto-changelog).
