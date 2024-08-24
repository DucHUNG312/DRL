#pragma once

#include "renderer/core.h"

namespace lab
{
namespace renderer
{
    template <typename... Args>
	inline std::string string_printf(const char* fmt, Args &&...args);

	// helpers, fwiw
	template <typename T>
	static auto operator<<(std::ostream& os, const T& v) -> decltype(v.to_string(), os)
	{
		return os << v.to_string();
	}
	template <typename T>
	static auto operator<<(std::ostream& os, const T& v) -> decltype(to_string(v), os)
	{
		return os << to_string(v);
	}

	template <typename T>
	inline std::ostream& operator<<(std::ostream& os, const std::shared_ptr<T>& p)
	{
		if (p)
			return os << p->to_string();
		else
			return os << "(nullptr)";
	}

	template <typename T>
	inline std::ostream& operator<<(std::ostream& os, const std::unique_ptr<T>& p)
	{
		if (p)
			return os << p->to_string();
		else
			return os << "(nullptr)";
	}

	namespace internal
	{
		std::string float_to_string(float v);
		std::string double_to_string(double v);

		template <typename T>
		struct IntegerFormatTrait;

		template <>
		struct IntegerFormatTrait<bool>
		{
			static constexpr const char* fmt() { return "d"; }
		};
		template <>
		struct IntegerFormatTrait<char>
		{
			static constexpr const char* fmt() { return "d"; }
		};
		template <>
		struct IntegerFormatTrait<unsigned char>
		{
			static constexpr const char* fmt() { return "d"; }
		};
		template <>
		struct IntegerFormatTrait<int>
		{
			static constexpr const char* fmt() { return "d"; }
		};
		template <>
		struct IntegerFormatTrait<unsigned int>
		{
			static constexpr const char* fmt() { return "u"; }
		};
		template <>
		struct IntegerFormatTrait<short>
		{
			static constexpr const char* fmt() { return "d"; }
		};
		template <>
		struct IntegerFormatTrait<unsigned short>
		{
			static constexpr const char* fmt() { return "u"; }
		};
		template <>
		struct IntegerFormatTrait<long>
		{
			static constexpr const char* fmt() { return "ld"; }
		};
		template <>
		struct IntegerFormatTrait<unsigned long>
		{
			static constexpr const char* fmt() { return "lu"; }
		};

		template <typename T>
		using HasSize =
			std::is_integral<typename std::decay_t<decltype(std::declval<T&>().size())>>;

		template <typename T>
		using HasData =
			std::is_pointer<typename std::decay_t<decltype(std::declval<T&>().data())>>;

		// Don't use size()/data()-based operator<< for std::string...
		inline std::ostream& operator<<(std::ostream& os, const std::string& str) {
			return std::operator<<(os, str);
		}

		template <typename T>
		inline std::enable_if_t<HasSize<T>::value&& HasData<T>::value, std::ostream&>
			operator<<(std::ostream& os, const T& v)
		{
			os << "[ ";
			auto ptr = v.data();
			for (size_t i = 0; i < v.size(); ++i)
			{
				os << ptr[i];
				if (i < v.size() - 1)
					os << ", ";
			}
			return os << " ]";
		}

		// base case
		void string_printf_recursive(std::string* s, const char* fmt);

		// 1. Copy from fmt to *s, up to the next formatting directive.
		// 2. Advance fmt past the next formatting directive and return the
		//    formatting directive as a string.
		std::string copy_to_format_string(const char** fmt_ptr, std::string* s);

		template <typename T>
		inline typename std::enable_if_t<!std::is_class_v<typename std::decay_t<T>>, std::string> format_one(const char* fmt, T&& v) 
        {
			// Figure out how much space we need to allocate; add an extra
			// character for the '\0'.
			size_t size = snprintf(nullptr, 0, fmt, v) + 1;
			std::string str;
			str.resize(size);
			snprintf(&str[0], size, fmt, v);
			str.pop_back();  // remove trailing NUL
			return str;
		}

		template <typename T>
		inline typename std::enable_if_t<std::is_class_v<typename std::decay_t<T>>, std::string> format_one(const char* fmt, T&& v)
		{
			LAB_LOG_FATAL("Printf: Non-basic type %s passed for format string %s", typeid(v).name(), fmt);
			return "";
		}

		template <typename T, typename... Args>
		inline void string_printf_recursive(std::string* s, const char* fmt, T&& v, Args &&...args);

		template <typename T, typename... Args>
		inline void string_printf_recursive_with_precision(std::string* s, const char* fmt, const std::string& nextFmt, T&& v, Args &&...args)
		{
			LAB_LOG_FATAL("MEH");
		}

		template <typename T, typename... Args>
		inline typename std::enable_if_t<!std::is_class_v<typename std::decay_t<T>>, void> string_printf_recursive_with_precision(std::string* s, const char* fmt, const std::string& nextFmt, int precision, T&& v, Args &&...args)
		{
			size_t size = snprintf(nullptr, 0, nextFmt.c_str(), precision, v) + 1;
			std::string str;
			str.resize(size);
			snprintf(&str[0], size, nextFmt.c_str(), precision, v);
			str.pop_back();  // remove trailing NUL
			*s += str;
			string_printf_recursive(s, fmt, std::forward<Args>(args)...);
		}

		// General-purpose version of string_printf_recursive; add the formatted
		// output for a single string_printf() argument to the final result string
		// in *s.
		template <typename T, typename... Args>
		inline void string_printf_recursive(std::string* s, const char* fmt, T&& v, Args &&...args)
		{
			std::string nextFmt = copy_to_format_string(&fmt, s);
			bool precisionViaArg = nextFmt.find('*') != std::string::npos;

			bool isSFmt = nextFmt.find('s') != std::string::npos;
			bool isDFmt = nextFmt.find('d') != std::string::npos;

			if constexpr (std::is_integral_v<std::decay_t<T>>)
			{
				if (precisionViaArg)
				{
					string_printf_recursive_with_precision(s, fmt, nextFmt, v,
						std::forward<Args>(args)...);
					return;
				}
			}
			else if (precisionViaArg)
				LAB_LOG_FATAL("Non-integral type provided for %* format.");

			if constexpr (std::is_same_v<std::decay_t<T>, float>)
				if (nextFmt == "%f" || nextFmt == "%s")
				{
					*s += internal::float_to_string(v);
					goto done;
				}

			if constexpr (std::is_same_v<std::decay_t<T>, double>)
				if (nextFmt == "%f" || nextFmt == "%s")
				{
					*s += internal::double_to_string(v);
					goto done;
				}

			if constexpr (std::is_same_v<std::decay_t<T>, bool>)  // FIXME: %-10s with bool
				if (isSFmt)
				{
					*s += bool(v) ? "true" : "false";
					goto done;
				}

			if constexpr (std::is_integral_v<std::decay_t<T>>)
			{
				if (isDFmt)
				{
					nextFmt.replace(nextFmt.find('d'), 1,
						internal::IntegerFormatTrait<std::decay_t<T>>::fmt());
					*s += format_one(nextFmt.c_str(), std::forward<T>(v));
					goto done;
				}
			}
			else if (isDFmt)
				LAB_LOG_FATAL("Non-integral type passed to %d format.");

			if (isSFmt)
			{
				std::stringstream ss;
				ss << v;
				*s += format_one(nextFmt.c_str(), ss.str().c_str());
			}
			else if (!nextFmt.empty())
				*s += format_one(nextFmt.c_str(), std::forward<T>(v));
			else
				LAB_LOG_FATAL("Excess values passed to Printf.");
		done:
			string_printf_recursive(s, fmt, std::forward<Args>(args)...);
		}

	} 

	// Printing Function Declarations
	template <typename... Args>
	void Printf(const char* fmt, Args &&...args);
	template <typename... Args>
	inline std::string string_printf(const char* fmt, Args &&...args);

	template <typename... Args>
	inline std::string string_printf(const char* fmt, Args &&...args)
	{
		std::string ret;
		internal::string_printf_recursive(&ret, fmt, std::forward<Args>(args)...);
		return ret;
	}

	template <typename... Args>
	void Printf(const char* fmt, Args &&...args)
	{
		std::string s = string_printf(fmt, std::forward<Args>(args)...);
		fputs(s.c_str(), stdout);
	}

#ifdef __GNUG__
#pragma GCC diagnostic pop
#endif  // __GNUG__

	// https://en.wikipedia.org/wiki/ANSI_escape_code#Colors
	inline std::string red(const std::string& s)
	{
		const char* red = "\033[1m\033[31m";  // bold red
		const char* reset = "\033[0m";
		return std::string(red) + s + std::string(reset);
	}

	inline std::string yellow(const std::string& s)
	{
		// https://en.wikipedia.org/wiki/ANSI_escape_code#8-bit
		const char* yellow = "\033[1m\033[38;5;100m";
		const char* reset = "\033[0m";
		return std::string(yellow) + s + std::string(reset);
	}

	inline std::string green(const std::string& s)
	{
		// https://en.wikipedia.org/wiki/ANSI_escape_code#8-bit
		const char* green = "\033[1m\033[38;5;22m";
		const char* reset = "\033[0m";
		return std::string(green) + s + std::string(reset);
	}

}

}