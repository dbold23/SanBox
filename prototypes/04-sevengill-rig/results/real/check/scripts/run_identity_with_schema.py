"""Zero-modification driver: texture_identity.run() with schema_path pointed at this
checkout's keypoints_sevengill_v1.yaml (the CLI has no --schema flag and
pattern.DEFAULT_SCHEMA_PATH is hardcoded to /home/user/SanBox/...). Everything else
is exactly what texture_identity.main() passes."""
import os, sys, warnings
warnings.simplefilter("always")
HERE = os.environ["RIG_DIR"]
sys.path.insert(0, HERE)
os.chdir(HERE)
import texture_identity as ti
schema = os.path.join(os.environ["WT"], "phase1b", "p0-sevengill-schema", "keypoints_sevengill_v1.yaml")
assert os.path.isfile(schema), schema
print("schema_path =", schema, flush=True)
ti.run(
    glb="assets/sevengill.glb", out_dir="results/real/identity",
    n_resights=4, years=3.0, n_random=3, seed=0, n_stations=64,
    tex_size=ti.DEFAULT_TEX_SIZE, chart_shape=ti.default_chart_shape(ti.CHART_H_PHI),
    validate=True, report=True, schema_path=schema,
)
