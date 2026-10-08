from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest


SOURCE = Path(__file__).resolve().parents[1] / "installer.cc"


def function(source, signature):
  start = source.index(signature)
  opening = source.index("{", start)
  depth = 1
  end = opening + 1
  while depth:
    depth += (source[end] == "{") - (source[end] == "}")
    end += 1
  return source[start:end]


class TestInstallerFailure(unittest.TestCase):
  def test_native_install_failure_paths(self):
    compiler = shutil.which("clang++") or shutil.which("g++")
    if compiler is None:
      self.skipTest("C++ compiler unavailable")
    source = SOURCE.read_text(encoding="utf-8")
    functions = "\n".join(function(source, signature) for signature in
                          ("void run(", "void cloneFinished("))
    functions = functions.replace("std::system(", "test_system(").replace("std::rename(", "test_rename(")
    functions = functions.replace("chdir(", "test_chdir(").replace("fopen(", "test_fopen(")
    harness = r"""
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <vector>
#define LOGD(...) ((void)0)
#define LOGE(...) ((void)0)
#define TMP_INSTALL_PATH "/test/new"
#define CONTINUE_PATH "/test/continue"
const std::string INSTALL_PATH = "/test/current", BACKUP_INSTALL_PATH = "/test/backup", VALID_CACHE_PATH = "/test/cache";
std::string migrated_branch = "v23";
const unsigned char str_continue[] = "test";
const unsigned char *str_continue_end = str_continue + 4;
bool previous_install = true;
namespace util {
template<typename... Args> std::string string_format(const char *format, Args... args) {
  char buffer[1024]; snprintf(buffer, sizeof(buffer), format, args...); return buffer;
}
bool file_exists(const std::string &) { return previous_install; }
}
std::vector<std::string> commands, renames;
std::string failed_command;
int failed_rename = 0;
bool finished = false;
bool fail_chdir = false;
bool fail_script = false;
int progress = 0;
int test_system(const char *cmd) {
  commands.emplace_back(cmd);
  return !failed_command.empty() && commands.back().find(failed_command) != std::string::npos ? 1 : 0;
}
int test_chdir(const char *) { return fail_chdir ? -1 : 0; }
FILE *test_fopen(const char *, const char *) { return fail_script ? nullptr : tmpfile(); }
int test_rename(const char *from, const char *to) {
  renames.emplace_back(std::string(from) + " -> " + to);
  return failed_rename == static_cast<int>(renames.size()) ? -1 : 0;
}
void renderProgress(int value) { progress = value; }
void finishInstall() { finished = true; }
"""
    harness += functions
    harness += r"""
void reset() {
  commands.clear(); renames.clear(); failed_command.clear(); failed_rename=0; finished=false; progress=0;
  previous_install=true; fail_chdir=false; fail_script=false;
}
int main() {
  bool failed = false;
  try { cloneFinished(1); } catch (const std::runtime_error &) { failed=true; }
  assert(failed && commands.empty() && renames.empty() && !finished);
  for (const auto &cmd : {"git checkout", "git reset", "git submodule"}) {
    reset(); failed_command=cmd; failed=false;
    try { cloneFinished(0); } catch (const std::runtime_error &) { failed=true; }
    assert(failed && renames.empty() && !finished && progress != 100);
  }
  reset(); failed_rename=1; failed=false;
  try { cloneFinished(0); } catch (const std::runtime_error &) { failed=true; }
  assert(failed && renames.size()==1 && !finished);
  reset(); failed_rename=2; failed=false;
  try { cloneFinished(0); } catch (const std::runtime_error &) { failed=true; }
  assert(failed && renames.size()==3 && renames.back()=="/test/backup -> /test/current" && !finished);
  reset(); fail_chdir=true; failed=false;
  try { cloneFinished(0); } catch (const std::runtime_error &) { failed=true; }
  assert(failed && commands.empty() && renames.empty() && !finished);
  for (const auto &cmd : {"chmod +x", "mv /data/continue"}) {
    reset(); failed_command=cmd; failed=false;
    try { cloneFinished(0); } catch (const std::runtime_error &) { failed=true; }
    assert(failed && renames.size()==2 && commands.back().find(cmd)!=std::string::npos && !finished && progress!=100);
  }
  reset(); fail_script=true; failed=false;
  try { cloneFinished(0); } catch (const std::runtime_error &) { failed=true; }
  assert(failed && renames.size()==2 && !finished && commands.back()!="rm -rf /test/backup");
  reset(); cloneFinished(0);
  assert(finished && progress==100 && renames.size()==2);
  assert(commands[2]=="git submodule update --init --recursive");
  assert(commands.back()=="rm -rf /test/backup");
  reset(); previous_install=false; cloneFinished(0);
  assert(finished && renames.size()==1 && renames.front()=="/test/new -> /test/current");
  assert(commands.back()=="mv /data/continue.sh.new /test/continue");
  puts("Installer clone/checkout/reset/submodule failures, replacement rollback and success passed");
}
"""
    with tempfile.TemporaryDirectory() as directory:
      cpp = Path(directory) / "installer_test.cc"
      executable = Path(directory) / "installer_test.exe"
      cpp.write_text(harness, encoding="utf-8")
      result = subprocess.run([compiler, "-std=c++17", str(cpp), "-o", str(executable)], capture_output=True, text=True)
      self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
      result = subprocess.run([str(executable)], capture_output=True, text=True)
      self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


if __name__ == "__main__":
  unittest.main()
