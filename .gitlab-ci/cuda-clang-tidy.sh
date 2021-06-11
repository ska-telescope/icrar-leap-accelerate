find compile_commands.json -type f -exec sed -i 's/ -x cu / -x cuda /g' {} \;
python -u /usr/bin/run-clang-tidy-11.py -quiet 2>&1 | tee clang-tidy.log
