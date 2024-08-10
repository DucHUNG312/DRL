#pragma once

#include "core.h"

#define SPDLOG_EOL ""
#include <spdlog/spdlog.h>
#include <spdlog/fmt/ostr.h>

namespace lab
{
class Logger
{
public:
	static void init();
	static void shutdown();
	static std::shared_ptr<spdlog::logger>& get_core_logger() { return core_logger; }
	static void add_sink(std::shared_ptr<spdlog::sinks::sink>& sink);
private:
	static std::shared_ptr<spdlog::logger> core_logger;
};
}