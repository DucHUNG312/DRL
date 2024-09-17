#pragma once

#include <cstdint>
#include <string>

namespace lab {
class Logger {
 public:
  enum class Severity : int8_t { FATAL = 0, ERROR, WARNING, INFO, DEBUG, VERBOSE };

 private:
  Severity severity_;
  bool color_;

 public:
  Logger(Severity severity = Severity::DEBUG, bool color = true);
  void log(Severity severity, const std::string& msg) noexcept;
  Severity get_reportable_severity() const noexcept;
  void set_reportable_log_severity(Severity severity) noexcept;
  void set_log_color(bool color) noexcept;
};

Logger& get_logger() noexcept;
} // namespace lab