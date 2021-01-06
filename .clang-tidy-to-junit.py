#!/usr/bin/env python3

# MIT License

# Copyright (c) 2018 PSPDFKit

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import sys
import collections
import re
import logging
import itertools
from typing import List
from xml.sax.saxutils import escape

def main():
    if len(sys.argv) < 2:
        logging.error("Usage: %s base-filename-path", sys.argv[0])
        logging.error(
            "  base-filename-path: Removed from the filenames to make nicer paths.")
        sys.exit(1)
    converter = ClangTidyConverter(sys.argv[1])
    converter.convert(sys.stdin, sys.stdout)

# Create a `ErrorDescription` tuple with all the information we want to keep.
ErrorDescription = collections.namedtuple(
    'ErrorDescription', 'file line column severity error error_identifier description')

FailureDescription = collections.namedtuple(
    'FailureDescription', 'file line column severity error error_identifier description')

class ClangTidyConverter:
    # All the errors encountered.
    errors: List[ErrorDescription] = []
    failures: List[ErrorDescription] = []

    # Parses the error.
    # Group 1: file path
    # Group 2: line
    # Group 3: column
    # Group 4: severity
    # Group 5: error message
    # Group 6: error identifier
    error_regex = re.compile(
        r"^([\w\/\.\-\ ]+):(\d+):(\d+): ([a-z]+): (.+) (\[[\w\-,\.]+\])$"
    )

    # Group 1: failure
    failure_regex = re.compile(
        r"[\w\/\.\-]+"
    )

    # This identifies the main error line (it has a [the-warning-type] at the end)
    # We only create a new error when we encounter one of these.
    main_error_identifier = re.compile(r'\[[\w\-,\.]+\]$')

    main_note_identifier = re.compile(r'\[[\w\-,\.]+\]$')

    main_failure_identifier = re.compile(r'^Error while processing ')

    def __init__(self, basename):
        self.basename = basename

    def print_junit_errors(self, sorted_errors, output_file):
        """
        Prints errors into test suites
        """
        # Iterate through the errors, grouped by file.
        for file, errorIterator in itertools.groupby(sorted_errors, key=lambda x: x.file):
            errors = list(errorIterator)
            error_count = len(errors)

            # Each file gets a test-suite
            output_file.write("""    <testsuite errors="{error_count}" name="{file}" tests="{error_count}" failures="0" time="0">\n"""
                              .format(error_count=error_count, file=file))
            for error in errors:
                # Write each error as a test case.
                output_file.write("""\
        <testcase name="{name}" time="0">
            <error type="{severity}" line="{line}" column="{column}" message="{message}">\n{htmldata}\n            </error>
        </testcase>\n""".format(type=escape(""),
                              line=error.line,
                              column=error.column,
                              severity=error.severity,
                              name=escape(error.error_identifier),
                              message=escape(error.error, entities={"\"": "&quot;"}),
                              htmldata=escape(error.description)))
            output_file.write("    </testsuite>\n")

    def print_junit_failures(self, sorted_failures, output_file):
        """
        Prints failures into test suites
        """
        # Iterate through the errors, grouped by file.
        for file, errorIterator in itertools.groupby(sorted_failures, key=lambda x: x.file):
            failures = list(errorIterator)
            failure_count = len(failures)

            # Each file gets a test-suite
            output_file.write("""    <testsuite errors="0" name="{file}" tests="{failure_count}" failures="{failure_count}" time="0">\n"""
                              .format(failure_count=failure_count, file=file))
            for failure in failures:
                # Write each error as a test case.
                output_file.write("""\
        <testcase name="{name}" time="0">
            <failure type="{severity}">\n{htmldata}\n            </failure>
        </testcase>\n""".format(type=escape(""),
                              severity=failure.severity,
                              name=escape(failure.error_identifier),
                              htmldata=escape(failure.description)))
            output_file.write("    </testsuite>\n")

    def print_junit_file(self, output_file):
        # Write the header.
        output_file.write("""<?xml version="1.0" encoding="UTF-8" ?>
<testsuites id="1" name="Clang-Tidy" tests="{test_count}" errors="{error_count}" failures="{failure_count}" time="0">\n"""
            .format(
                test_count=len(self.errors)+len(self.failures),
                error_count=len(self.errors),
                failure_count=len(self.failures)))

        sorted_errors = sorted(self.errors, key=lambda x: x.file)
        self.print_junit_failures(self.failures, output_file)
        self.print_junit_errors(sorted_errors, output_file)
        output_file.write("</testsuites>\n")

    def process_error(self, error_array: List[str]):
        """
        Processes raw error text into this object's error collection
        """
        if len(error_array) == 0:
            return

        if self.main_error_identifier.search(error_array[0], re.M):
            # Processing an error
            result = self.error_regex.match(error_array[0])
            if result is None:
                logging.warning(
                    'Could not match error_array to regex: %s', error_array)
                return

            # We remove the `basename` from the `file_path` to make prettier filenames in the JUnit file.
            file_path = result.group(1).replace(self.basename, "")
            error = ErrorDescription(
                file_path,
                int(result.group(2)),
                int(result.group(3)),
                result.group(4),
                result.group(5),
                result.group(6),
                "".join(error_array[1:]).rstrip())
            self.errors.append(error)

        elif self.main_failure_identifier.search(error_array[0]):
            #Processing a failure
            result = self.failure_regex.match(error_array[0])
            if result is None:
                logging.warning(
                    'Could not match error_array to regex: %s', error_array)
                return
            error = ErrorDescription(
                result.group(0),
                0,
                0,
                "failure",
                "identifier",
                "message",
                "".join(error_array).rstrip())
            self.failures.append(error)

    def convert(self, input_file, output_file):
        # Collect all lines related to one error.
        current_error = []
        error_ended = True
        for line in input_file:
            # If the line starts with a `/`, it is the start line of an error about a file.
            # If the line starts with "Error while processing ", a linting failure about a file has occured.
            if line[0] == '/' and self.main_error_identifier.search(line, re.M):
                # Start of an error. Process any existing `current_error` we might have
                self.process_error(current_error)
                # Initialize `current_error` with the first line of the error.
                current_error = [line]
                error_ended = False
            elif "Error while processing " in line:
               # Start of a failure
               self.process_error(current_error)
               # Initialize `current_error` with the first line of the error.
               current_error = [line]
               error_ended = False

            elif len(current_error) > 0:
                if "in non-user code" in line or " warnings generated." in line or "clang-tidy" in line:
                    # Ignore output if error has ended
                    error_ended = True

                # If the line didn't start with a `/` and we have a `current_error`, we simply append
                # the line as additional information.
                if not error_ended:
                    current_error.append(line)
            else:
                pass

        # If we still have any current_error after we read all the lines,
        # process it.
        if len(current_error) > 0:
            self.process_error(current_error)

        # Print the junit file.
        self.print_junit_file(output_file)


if __name__ == "__main__":
    main()
