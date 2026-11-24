"""patch_deadline.py SPIRAL_DIR : add FIT_SPIRAL_LOOP_SECONDS to fit_spiral.py (after patch_fitter.py), idempotently.
When set, the training loop stops cleanly once that many seconds have passed since the loop started; the checkpoint
and meshes are written with completed_iterations = the last finished step, so a later run resumes the same schedule."""
import sys, pathlib
p = pathlib.Path(sys.argv[1]) / 'fit_spiral.py'; s = p.read_text()
MARK = '# FIT_SPIRAL_LOOP_SECONDS (sp lane)'
if MARK in s: print('already patched'); sys.exit(0)
old = """            loss, losses, log_metrics, shell_metrics = self.step(iteration)
            progress.update(iteration - self.start_iteration + 1)"""
new = """            """ + MARK + """
            if iteration == self.start_iteration:
                self._sp_t0 = time.time(); self._sp_budget = float(os.environ.get('FIT_SPIRAL_LOOP_SECONDS', '0') or 0)
            if self._sp_budget > 0 and time.time() - self._sp_t0 > self._sp_budget:
                print(f'FIT_SPIRAL_LOOP_SECONDS reached: stopping before iteration {iteration}', flush=True)
                self._loop_end = iteration
                break
            loss, losses, log_metrics, shell_metrics = self.step(iteration)
            progress.update(iteration - self.start_iteration + 1)"""
if s.count(old) != 1: sys.exit(f'anchor count {s.count(old)}')
p.write_text(s.replace(old, new)); print('patched FIT_SPIRAL_LOOP_SECONDS')
