#include "lab/utils/logger.h"

#include <spdlog/sinks/rotating_file_sink.h>
#include <spdlog/sinks/stdout_color_sinks.h>

namespace lab
{
namespace utils
{
std::shared_ptr<spdlog::logger> Logger::core_logger;
std::vector<spdlog::sink_ptr> sinks;

void Logger::init()
{
	sinks.emplace_back(std::make_shared<spdlog::sinks::stdout_color_sink_mt>()); // debug
	// sinks.emplace_back(std::make_shared<ImGuiConsoleSink_mt>()); // ImGuiConsole

	auto logFileSink = std::make_shared<spdlog::sinks::rotating_file_sink_mt>("lab.log", 1048576 * 5, 3);
	sinks.emplace_back(logFileSink); // Log file
	// create the loggers
	core_logger = std::make_shared<spdlog::logger>("lab", begin(sinks), end(sinks));
	spdlog::register_logger(core_logger);

	// configure the loggers
#ifdef LOG_TIMESTAMP
	spdlog::set_pattern("%^[%T] %v%$");
#else
	spdlog::set_pattern("%v%$");
#endif // LOG_TIMESTAMP
	core_logger->set_level(spdlog::level::trace);
}

void Logger::shutdown()
{
	core_logger.reset();
	spdlog::shutdown();
}

void Logger::add_sink(spdlog::sink_ptr& sink)
{
	core_logger->sinks().push_back(sink);
	core_logger->set_pattern("%v%$");
}
}
}