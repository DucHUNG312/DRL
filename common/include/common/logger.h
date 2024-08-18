#pragma once

#define SPDLOG_EOL ""
#ifdef __GNUC__
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunknown-pragmas"
#pragma GCC diagnostic ignored "-Wstringop-overflow"
#endif
#include <spdlog/spdlog.h>
#include <spdlog/fmt/ostr.h>
#ifdef __GNUC__
#pragma GCC diagnostic pop
#endif

namespace lab
{
namespace common
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
}