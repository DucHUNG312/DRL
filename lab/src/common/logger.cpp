#include "lab/common/logger.h"

#include <iostream>
#include <string>

#define TERM_NORMAL "\033[0m";
#define TERM_RED "\033[0;31m";
#define TERM_YELLOW "\033[0;33m";
#define TERM_GREEN "\033[0;32m";
#define TERM_MAGENTA "\033[1;35m";

namespace lab {

Logger::Logger(Severity severity /*= Severity::DEBUG*/, bool color /*= true*/) : severity_(severity), color_(color) {}

void Logger::log(Severity severity, const std::string& msg) noexcept {
  if (severity > severity_) {
    return;
  }

  if (color_) {
    switch (severity_) {
      case Severity::FATAL:
        std::cerr << TERM_RED;
        break;
      case Severity::ERROR:
        std::cerr << TERM_RED;
        break;
      case Severity::WARNING:
        std::cerr << TERM_YELLOW;
        break;
      case Severity::INFO:
        std::cerr << TERM_GREEN;
        break;
      case Severity::DEBUG:
        std::cerr << TERM_MAGENTA;
        break;
      case Severity::VERBOSE:
        std::cerr << TERM_NORMAL;
        break;
      default:
        break;
    }
  }

  switch (severity_) {
    case Severity::FATAL:
      std::cerr << "FATAL: ";
      break;
    case Severity::ERROR:
      std::cerr << "ERROR: ";
      break;
    case Severity::WARNING:
      std::cerr << "WARNING: ";
      break;
    case Severity::INFO:
      std::cerr << "INFO: ";
      break;
    case Severity::DEBUG:
      std::cerr << "DEBUG: ";
      break;
    case Severity::VERBOSE:
      std::cerr << "VERBOSE: ";
      break;
    default:
      std::cerr << "UNKNOWN: ";
      break;
  }

  if (color_) {
    std::cerr << TERM_NORMAL;
  }

  std::cerr << msg << '\n';
}

Logger::Severity Logger::get_reportable_severity() const noexcept {
  return severity_;
}

void Logger::set_reportable_log_severity(Severity severity) noexcept {
  severity_ = severity;
}

void Logger::set_log_color(bool color) noexcept {
  color_ = color;
}

Logger& get_logger() noexcept {
  static Logger logger{Logger::Severity::DEBUG, true};
  return logger;
}

} // namespace lab