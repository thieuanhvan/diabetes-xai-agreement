Cách dùng:

Bây giờ (19/05): Extract zip vào diabetes-xai-agreement/_post_mapr_cleanup/ hoặc tương đương trong local repo. Folder này không commit vào git public (add to .gitignore hoặc keep private branch).
Khi MAPR notif arrives (~30/06): Mở README.md → CHECKLIST.md → follow Path A (Accept) hoặc Path B (Reject).
Repo patches (repo_patches/ ~30 min total):

configs/default.yaml: 3 patch patterns sẵn (comment-only / key-only / both) — Van check current file structure rồi áp đúng pattern. Sanity check grep command included.
environment.json: template với lower-bound versions; populate exact ==X.Y.Z từ pip freeze của environment đã chạy run 12. Audit-trail fields đã đầy đủ (AUC numbers, runtime hours, dataset constants, known issues).


Tier 3 analyses (~10-13h total compute, defer đến R1 revision hoặc journal extension):

plan.md: 3 analyses (multiseed × 5 / topK × 4 / PI n_repeats × 4), each with settings + expected outputs + reporting template + integration into main_vi v76 supplement
runner_template.py: skeleton Python — Van adapt imports cho repo layout (src/pipelines/ vs analysis/), uncomment actual logic. Helpers (jaccard) stub included.