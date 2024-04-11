# CHANGELOG



## v0.0.4 (2024-04-11)

### Ci

* ci: test badges on rtd index ([`71a1ff1`](https://github.com/LedgerInvesting/bayesblend/commit/71a1ff18a3d04f744db96085c72ba24dbb2345cf))

* ci: mypy, docs badges ([`d41e492`](https://github.com/LedgerInvesting/bayesblend/commit/d41e49249b42163875d9c468e15f7dc0c368e9a4))

* ci: ruff badge ([`e0c5a31`](https://github.com/LedgerInvesting/bayesblend/commit/e0c5a311e3626724adea243c30a05cfa77f5f789))

* ci: retry workflow badge ([`c96d0df`](https://github.com/LedgerInvesting/bayesblend/commit/c96d0dfc7443cc19da7dd9501445d839ca5c871b))

* ci: fix workflow badge ([`cb2d9e8`](https://github.com/LedgerInvesting/bayesblend/commit/cb2d9e860673f1fe5eb38faf1638b976f04f9adb))

* ci: add workflow badge to readme ([`db4634b`](https://github.com/LedgerInvesting/bayesblend/commit/db4634b91068f7bb296512dcd23b521095ba1d7c))

### Documentation

* docs: clarify model averaging throughout

We used the term &#34;Bayesian model averaging&#34; before, but realize that this could be a bit misleading in that we do stacking and pseudo-BMA. Removing the BMA terminology from docs to not confuse users. ([`2640217`](https://github.com/LedgerInvesting/bayesblend/commit/2640217cc07a1c587477526e31e3c9b083d7e403))

### Fix

* fix(io): name of arrays sent to Draws

io.Draws.from_* methods were previously using the user-specified names of log_lik and post_pred arrays instead of the stadardized names when initializing Draws objects. This produced errors when, e.g., the log_lik array was named something different (e.g., loglik) from the standard name. ([`a5ae9c9`](https://github.com/LedgerInvesting/bayesblend/commit/a5ae9c9a2c2c0654383d8eabc00c0ebb0fdfd421))

### Refactor

* refactor(io): simplify access to stan variables ([`5b3dd11`](https://github.com/LedgerInvesting/bayesblend/commit/5b3dd1184a757807df4e9f3c94c28909879e3449))

### Test

* test(io): add test for different posterior array names ([`2293e8c`](https://github.com/LedgerInvesting/bayesblend/commit/2293e8cb8e1ed92b1e540818f968ebc17b2cf125))

### Unknown

* Merge pull request #30 from LedgerInvesting/fix-io-array-names

fix(io): name of arrays sent to Draws ([`f7317db`](https://github.com/LedgerInvesting/bayesblend/commit/f7317db86cce28ce5463ae72409c2473e117a308))

* Merge pull request #27 from LedgerInvesting/docs-clarify-averaging

docs: clarify model averaging throughout ([`d058bf8`](https://github.com/LedgerInvesting/bayesblend/commit/d058bf8a6552cc82a0ea44330cb6f2b92d82b0a7))

* Merge pull request #25 from LedgerInvesting/ci-badges

ci: add badges ([`bb5795a`](https://github.com/LedgerInvesting/bayesblend/commit/bb5795a96bd82dec1ebc40982d5fa5c74877c83e))

* Merge pull request #24 from LedgerInvesting/docs-fix-readme-links

fix: fix documentation README links ([`c395191`](https://github.com/LedgerInvesting/bayesblend/commit/c3951912520e124b551f7172ffd6018924338a5b))


## v0.0.3 (2024-04-08)

### Documentation

* docs: update readme links to latest ([`9b035b8`](https://github.com/LedgerInvesting/bayesblend/commit/9b035b8271598995599370f61d6b3984c383c3b6))

* docs: re-run mixture model with separate coefficients ([`da345c4`](https://github.com/LedgerInvesting/bayesblend/commit/da345c43cd313c175339cbedea5f4a5f76a59196))

### Fix

* fix: fix-readme-links ([`8808b65`](https://github.com/LedgerInvesting/bayesblend/commit/8808b657e4dea38ca6c53c63a1e196d0b939f053))

### Unknown

* Merge pull request #17 from LedgerInvesting/make-docs

`MkDocs` documentation ([`aecd31c`](https://github.com/LedgerInvesting/bayesblend/commit/aecd31c7f4cf546ec9439e4b6888e8eaa40c8e9f))


## v0.0.2 (2024-04-08)

### Documentation

* docs: Arviz -&gt; ArviZ ([`69bed7f`](https://github.com/LedgerInvesting/bayesblend/commit/69bed7f0b29e8bbbef40ae496a93de1967de830d))

* docs: Add Arviz integration user guid

Add examples of using `from_arviz`, `to_arviz`,
and `from_lpd` in the context of `arviz.InferenceData`
objects. ([`10284e7`](https://github.com/LedgerInvesting/bayesblend/commit/10284e73f6661346de1820e67d1b54323de5662d))

* docs: fix ruff ([`130a650`](https://github.com/LedgerInvesting/bayesblend/commit/130a650ee417d2bb410add581fe6a9565162e665))

* docs: ignore mypy errors in scripts ([`9fbee5f`](https://github.com/LedgerInvesting/bayesblend/commit/9fbee5f19c6db3c383a1ea7b24a18c8f483c2db9))

* docs: add figures in user-guide ([`549ac0a`](https://github.com/LedgerInvesting/bayesblend/commit/549ac0a9a97be07dac39216918c877152e4e172b))

* docs: add comparison vignette

Add a comparison notebook using simulated data.

Addresses #10 ([`16f12b8`](https://github.com/LedgerInvesting/bayesblend/commit/16f12b805d1872dc6453dd1f7a8bf09a307c6ce7))

### Fix

* fix: ruff ambiguous name ([`0ed0b2e`](https://github.com/LedgerInvesting/bayesblend/commit/0ed0b2e3ea105a738ad4f6e8a05ed85f203839f0))

* fix: Fix the creation of InferenceData objects in `to_arviz`

Fill the `log_likelihood` and `posterior_predictive` slots of
InferenceData directly, don&#39;t just pass a large dict. ([`3033f3b`](https://github.com/LedgerInvesting/bayesblend/commit/3033f3b3282e946479379e17e9c5df083c380e07))

* fix: add poetry.lock back in ([`9e8869b`](https://github.com/LedgerInvesting/bayesblend/commit/9e8869b2bef681cd370ba0f1b646eb4bdec2a4e2))

### Refactor

* refactor: remove unecessary comment ([`e7937a2`](https://github.com/LedgerInvesting/bayesblend/commit/e7937a240b42942c967d252c3fd14d91a70d4697))

### Test

* test: add missing import ([`4086bdf`](https://github.com/LedgerInvesting/bayesblend/commit/4086bdf5ea5c66f31ca19d06002f27070686d7c3))

* test: fix tests ([`022df8c`](https://github.com/LedgerInvesting/bayesblend/commit/022df8c022c9770d9774a826c477033e2f120a48))

* test: add test comment on mean/logmeanexp ([`ead3334`](https://github.com/LedgerInvesting/bayesblend/commit/ead3334d29cc3f43e93c3961f97d0aaad19fc1cc))

### Unknown

* Merge branch &#39;main&#39; of git+ssh://github.com/LedgerInvesting/bayesblend into make-docs ([`eef7772`](https://github.com/LedgerInvesting/bayesblend/commit/eef7772ea5bdd6cbd06fe3d2567751db6e6d8d73))

* Merge pull request #23 from LedgerInvesting/feature-arviz-wrappers

feature: add from/to arviz wrappers ([`3022a17`](https://github.com/LedgerInvesting/bayesblend/commit/3022a17cf62fdbaeb0acabdae17614d24552ac0a))

* Merge branch &#39;feature-arviz-wrappers&#39; of git+ssh://github.com/LedgerInvesting/bayesblend into make-docs ([`810ac4b`](https://github.com/LedgerInvesting/bayesblend/commit/810ac4b5f52f638013520d4901a59258fa44b522))

* feature: Create BayesBlendModel objects from `lpd` values

Some use cases of BayesBlend will be estimating the LPD values
first via a variety of algorithms (e.g. PSIS-LOO, cross-validation)
and then wanting to blend predictions. This adds a `from_lpd` method
to allow us to handle these cases. ([`31b3592`](https://github.com/LedgerInvesting/bayesblend/commit/31b3592aec8a7ccfffcab3fb8573f2bce05c0c77))

* feature: add from/to arviz wrappers ([`eb91897`](https://github.com/LedgerInvesting/bayesblend/commit/eb91897abae1e158e4c8928d062097be69688f48))

* Merge branch &#39;main&#39; of git+ssh://github.com/LedgerInvesting/bayesblend into make-docs ([`6c4abd1`](https://github.com/LedgerInvesting/bayesblend/commit/6c4abd1531deb439e94250fcbf470afbaf47b5e0))


## v0.0.1 (2024-04-05)

### Documentation

* docs: add contributing.md ([`13bf5fe`](https://github.com/LedgerInvesting/bayesblend/commit/13bf5fed20fcddab081bd628a0ac3eb50b785308))

* docs: update READMD ([`b9af3db`](https://github.com/LedgerInvesting/bayesblend/commit/b9af3db443706cec58acdd1cbe1ed4bb2c712dad))

### Fix

* fix: token ([`48e99c1`](https://github.com/LedgerInvesting/bayesblend/commit/48e99c129c717ffd1f6449f2d673d08f55ca9247))

### Refactor

* refactor: use predict method instead of blend ([`061a2a3`](https://github.com/LedgerInvesting/bayesblend/commit/061a2a3e764b4979ebfa076b0d6ac3a812a322b2))

### Unknown

* Merge pull request #20 from LedgerInvesting/feature-ci-version-release

Minor: ci version release ([`3b2d8c6`](https://github.com/LedgerInvesting/bayesblend/commit/3b2d8c66027296517322c00a2fc26b3be87b0b2a))

* configure light/dark mode switch ([`eefca59`](https://github.com/LedgerInvesting/bayesblend/commit/eefca5998b64a93210aae634384b0adbf3519f9c))

* fix workflow ([`495889e`](https://github.com/LedgerInvesting/bayesblend/commit/495889eb7452e0f030c2b49bab309e70a4f7a872))

* fix tests ([`3c5b494`](https://github.com/LedgerInvesting/bayesblend/commit/3c5b4940bd69cc67c726cf49f95b9e5007749dbc))

* update RTD yaml ([`a3d654a`](https://github.com/LedgerInvesting/bayesblend/commit/a3d654a8bb6933253eb31574b04bbd9732ef98cd))

* update RTD yaml ([`7a52f26`](https://github.com/LedgerInvesting/bayesblend/commit/7a52f26283ad49612435f30d5cf65909ab36af16))

* update RTD yaml ([`909698a`](https://github.com/LedgerInvesting/bayesblend/commit/909698a83d020c021faf01a0dfcb1ba6d5f5dd4e))

* test build ([`0af6e6b`](https://github.com/LedgerInvesting/bayesblend/commit/0af6e6b44b5e31636126eff07a70d7a7f8ab14bc))

* install poetry in RTD ([`f9839c0`](https://github.com/LedgerInvesting/bayesblend/commit/f9839c05b3ab670411b41a9a4f52bf6b8342622b))

* test pipx in test again ([`b43b2e3`](https://github.com/LedgerInvesting/bayesblend/commit/b43b2e37808023683a49d4b74666c899e706b2a6))

* install poetry with 3.11 ([`83f86b1`](https://github.com/LedgerInvesting/bayesblend/commit/83f86b12f452ad121f8c570fe73e1236a991e115))

* update contributing docs ([`4850819`](https://github.com/LedgerInvesting/bayesblend/commit/4850819dc183d1afd00b3a6af469c5f3b67ad25f))

* fix ruff ([`31128c7`](https://github.com/LedgerInvesting/bayesblend/commit/31128c7a3dba5f0bf073601118703e259e98b25b))

* permissions in workflow ([`b30b91d`](https://github.com/LedgerInvesting/bayesblend/commit/b30b91d2086248db17b2f9921c40b7c1eb7745a6))

* update semantic release workflow ([`6c1d156`](https://github.com/LedgerInvesting/bayesblend/commit/6c1d156c4c7d72954ff08907c538b21a9125d64d))

* make MleStacking.fit fix the last weight ([`2231e0e`](https://github.com/LedgerInvesting/bayesblend/commit/2231e0edb2ca45967fc9914416b7ac630dcb4d2b))

* update readme ([`8361e10`](https://github.com/LedgerInvesting/bayesblend/commit/8361e10103d6d2fda271b36bde4ade782a0e6023))

* update readme ([`cd6b286`](https://github.com/LedgerInvesting/bayesblend/commit/cd6b28641929206080ee3df4e03d2df7dce68c93))

* Merge branch &#39;main&#39; of git+ssh://github.com/LedgerInvesting/bayesblend into make-docs ([`00f2565`](https://github.com/LedgerInvesting/bayesblend/commit/00f2565e28e72906340f240ea45606ad9bf706c0))

* add getting started, contributin, explanation of terms ([`936ad5e`](https://github.com/LedgerInvesting/bayesblend/commit/936ad5ecd527958cb3e1784ca39239c5eee4a123))

* move contributing to docs, remove from reop root ([`db4f16d`](https://github.com/LedgerInvesting/bayesblend/commit/db4f16d558b69bd50e4b543642f388c3068e89dd))

* expand allowable semantic types ([`29e4663`](https://github.com/LedgerInvesting/bayesblend/commit/29e4663ba914fe27a4f1b4cb48c1e37d7fff5235))

* order in workflow ([`0e2dd90`](https://github.com/LedgerInvesting/bayesblend/commit/0e2dd90808a92b7d2bb2c5d9a22a40f096c0b1c9))

* fix test workflow ([`c53f647`](https://github.com/LedgerInvesting/bayesblend/commit/c53f6477275dd05de115b15ae631b579e35921cf))

* minor: semantic versioning ([`7d5e188`](https://github.com/LedgerInvesting/bayesblend/commit/7d5e1880489e7d1e6b577684df2bc6a01bee9dc0))

* add license ([`f07428f`](https://github.com/LedgerInvesting/bayesblend/commit/f07428fc715ee0ebc6f704ad3ba61b27c693774d))

* monor: ci for semantic versioning and release ([`42799c2`](https://github.com/LedgerInvesting/bayesblend/commit/42799c2929e1dc1ae19dc029b9385e4c5315fb05))

* Merge pull request #14 from LedgerInvesting/refactor-blend-predict

refactor: use predict method instead of blend ([`4209a90`](https://github.com/LedgerInvesting/bayesblend/commit/4209a9024070750c508dfc153d7112771b991104))

* more informative docstring ([`7bf4e47`](https://github.com/LedgerInvesting/bayesblend/commit/7bf4e47deabd3462095407c16c1adbeb200e314a))

* update readme ([`97057d6`](https://github.com/LedgerInvesting/bayesblend/commit/97057d6ebaf2c0faf10582421168badd8a330602))

* better seed usage and update readme ([`b920a42`](https://github.com/LedgerInvesting/bayesblend/commit/b920a42fde9b424aae7ff32c73684fb98ea73ac6))

* fix tests ([`76689e7`](https://github.com/LedgerInvesting/bayesblend/commit/76689e733fadad29ae1ffabb1a3517785d0bd980))

* Merge branch &#39;main&#39; of git+ssh://github.com/LedgerInvesting/bayesblend into make-docs ([`a78ab6e`](https://github.com/LedgerInvesting/bayesblend/commit/a78ab6eb61780094f4c456d2b26402c2b514b12e))

* Merge pull request #16 from LedgerInvesting/contributing-suggestions

`CONTRIBUTING` and `README` suggestions ([`b94e0d7`](https://github.com/LedgerInvesting/bayesblend/commit/b94e0d7a050144fd9c868578e2ea008c589e5bec))

* update lock file ([`13e19ba`](https://github.com/LedgerInvesting/bayesblend/commit/13e19ba52e98c96fecedd15961ac28d180210f8d))

* add doc files and config ([`dfdca1a`](https://github.com/LedgerInvesting/bayesblend/commit/dfdca1ab5c878ea7ec6b5d7e1a456411cd6f18e1))

* minor grammar ([`15f1c2b`](https://github.com/LedgerInvesting/bayesblend/commit/15f1c2b00081b0e9314ddb1f8ed26d6f10f45be9))

* basic mkdocs setup ([`99859d9`](https://github.com/LedgerInvesting/bayesblend/commit/99859d9516fd8244f7c2ef45cff69e163c79867f))

* description ([`79bed6c`](https://github.com/LedgerInvesting/bayesblend/commit/79bed6c11cd83e2c5e81a613ad4f62685503dd96))

* propose sentence case for headings, camelcase for package naming ([`bfe788c`](https://github.com/LedgerInvesting/bayesblend/commit/bfe788ccaee811b5af5af98dae6dd2f25400a07e))

* pipx and python3.11 version for poetry ([`5e9ef7c`](https://github.com/LedgerInvesting/bayesblend/commit/5e9ef7cb9dde49417319dfc504b94763b6c372bf))

* Merge pull request #12 from LedgerInvesting/docs-contributing

docs: add contributing.md ([`47310e6`](https://github.com/LedgerInvesting/bayesblend/commit/47310e6810e1b325ba1e7ffc5054acd5dccb6735))

* Merge pull request #5 from LedgerInvesting/enhance-readme

docs: update readme ([`7a8ef54`](https://github.com/LedgerInvesting/bayesblend/commit/7a8ef541efa1c13ecc47585fa498800fbd8941e6))

* pass names along ([`df03f42`](https://github.com/LedgerInvesting/bayesblend/commit/df03f421372a8b7e548ca753c009116bfbb4dd56))

* Merge pull request #4 from LedgerInvesting/minor-add-io-draws-helper

minor: add io for help handling mcmc draws ([`a37ba08`](https://github.com/LedgerInvesting/bayesblend/commit/a37ba084889d62b25e42123d24522ca183663abb))

* again ([`b33fa06`](https://github.com/LedgerInvesting/bayesblend/commit/b33fa06bbebcd7d1976e7872844ad88ce43108bc))

* try again ([`d5ae4e4`](https://github.com/LedgerInvesting/bayesblend/commit/d5ae4e49e6c7f123d8df2dfabef11866024c134b))

* ignore test folder for gha checks ([`039056c`](https://github.com/LedgerInvesting/bayesblend/commit/039056cfddd7cdc07106bae19933e913ef7c249b))

* convenience shape chekcing method + tests ([`2dfcee0`](https://github.com/LedgerInvesting/bayesblend/commit/2dfcee00fb18bad822d205c6e3ff4e59976ac805))

* add draws docstring ([`3f4a0a6`](https://github.com/LedgerInvesting/bayesblend/commit/3f4a0a6d3be522c05047c27df01511c249ed7481))

* add draws class to models ([`40517f5`](https://github.com/LedgerInvesting/bayesblend/commit/40517f53e99cf3fdcc861bcecfb0e82a63a60338))

* smh^2 ([`9346ae5`](https://github.com/LedgerInvesting/bayesblend/commit/9346ae5f9d38ef31e863255f0ddf22e77e5d3b50))

* smh ([`3e63be4`](https://github.com/LedgerInvesting/bayesblend/commit/3e63be49c4ef77f1074c41419603510599eb7ce4))

* change gha fetch depth ([`657bba0`](https://github.com/LedgerInvesting/bayesblend/commit/657bba0b702a0fa331e4b774778a3bbf3f6a11db))

* fix name ([`67fcc8a`](https://github.com/LedgerInvesting/bayesblend/commit/67fcc8afbc4b5d7b2fe32f617300f1aaf4e8ce91))

* minor: add io for help handling mcmc draws ([`f861e99`](https://github.com/LedgerInvesting/bayesblend/commit/f861e999d3d7d28a9196f990feac4ba0a1ee33dd))

* Merge pull request #3 from LedgerInvesting/minor-add-ruff-linter

Minor add ruff linter ([`ff52894`](https://github.com/LedgerInvesting/bayesblend/commit/ff52894f140debc7fdaa017ee80e221d65da7b09))

* add black formatter to dev dependencies ([`ee9f15d`](https://github.com/LedgerInvesting/bayesblend/commit/ee9f15deea4d1fc091267fe5fa7e43bccb5c4693))

* change docstring to test ruff ([`3125e8e`](https://github.com/LedgerInvesting/bayesblend/commit/3125e8e91e0c4a1bb7a91c9f775f111e307389fb))

* add gha for ruff check ([`c41c975`](https://github.com/LedgerInvesting/bayesblend/commit/c41c9755a7bb84f27c2dbd28b40c95f395cc0a73))

* fix namees ([`c4dcf54`](https://github.com/LedgerInvesting/bayesblend/commit/c4dcf54c4abedbc1705faa7f436478fd94d7e9a1))

* get file change in separate task ([`c5729d4`](https://github.com/LedgerInvesting/bayesblend/commit/c5729d4760e6b6941f0918e8450ab025f5530ddf))

* rm unnecessary job ([`807d142`](https://github.com/LedgerInvesting/bayesblend/commit/807d14284a598f49128dfad2ef91110db4842666))

* add type checking to GHA ([`3a9eaed`](https://github.com/LedgerInvesting/bayesblend/commit/3a9eaed49ff7da6f597b2d9762720425f6408448))

* check if cmdstan cache hit ([`c628d53`](https://github.com/LedgerInvesting/bayesblend/commit/c628d53b7c2d15ecc61de51b07f384a10e1d76c9))

* rm flake8 ([`07064b4`](https://github.com/LedgerInvesting/bayesblend/commit/07064b49ed7e4b65b2f53b699dca840b5b40253f))

* minor: add ruff linter support, rm flake8 ([`e9d03b2`](https://github.com/LedgerInvesting/bayesblend/commit/e9d03b278519b7c0424f4600c22033522b185564))

* Merge pull request #2 from LedgerInvesting/minor-add-blend-method

Minor add blend method ([`a0da167`](https://github.com/LedgerInvesting/bayesblend/commit/a0da167c0da26e5672e077f6a751e1fc899ea717))

* minor: move blending to model method ([`3090053`](https://github.com/LedgerInvesting/bayesblend/commit/3090053fb97453e3fd037b393104275fab7be4b5))

* rm unnecessary weights file ([`e8a2650`](https://github.com/LedgerInvesting/bayesblend/commit/e8a26508b129de7f145dd8543b7336463f327fae))

* update poetry config ([`becb8d8`](https://github.com/LedgerInvesting/bayesblend/commit/becb8d8ef8446e84554cced5b5588c373a2cdb04))

* minor: move blend method to models ([`aaee6fb`](https://github.com/LedgerInvesting/bayesblend/commit/aaee6fb882482848300c058c2a56c2375a183011))

* Merge pull request #1 from LedgerInvesting/minor-ci-cd-tests

minor: add CI/CD for tests ([`0c7b494`](https://github.com/LedgerInvesting/bayesblend/commit/0c7b494a937ff0fd75b3a0b348ebbcf8aaf9bf02))

* smh ([`3b3d2fb`](https://github.com/LedgerInvesting/bayesblend/commit/3b3d2fbbf3e136b2340ddf704fc032dd124aedc9))

* udate pyproject ([`eef8e85`](https://github.com/LedgerInvesting/bayesblend/commit/eef8e85bca9a8ab421e9bcc4baab316c7005b81b))

* cmdstanpy install ([`e03d90b`](https://github.com/LedgerInvesting/bayesblend/commit/e03d90b83dcb3b452009db4438012b5a9fa2a731))

* update versions ([`6863a4f`](https://github.com/LedgerInvesting/bayesblend/commit/6863a4f5a070938514e4c9b7d54d9e6e955fdfe2))

* use poetry shell to install cmdstan ([`0494e36`](https://github.com/LedgerInvesting/bayesblend/commit/0494e36073c2128453f028cb3f669ddc7d22bfed))

* use poetry to install test dep ([`644afa9`](https://github.com/LedgerInvesting/bayesblend/commit/644afa95433efa01b134bf77d7ae19125888413e))

* rm login ([`afb8c0b`](https://github.com/LedgerInvesting/bayesblend/commit/afb8c0bc945f40aea3f74d531121f97260f7f52d))

* minor: add CI/CD for tests ([`44a4bab`](https://github.com/LedgerInvesting/bayesblend/commit/44a4bab377f3fdaf766654c2ff2b019d7ac3a75d))

* initial commit ([`4fb0093`](https://github.com/LedgerInvesting/bayesblend/commit/4fb00932a6218986669926241753a953e1a4730e))
