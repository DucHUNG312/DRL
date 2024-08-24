#pragma once

#include "renderer/core.h"

#include <uuid.h>

namespace lab
{

namespace renderer
{

// class UUID
// {
// public:
//     UUID(uint64_t seed = 0);
//     UUID(const UUID& other) = default;
//     UUID& operator=(const UUID& other) = default;
//     UUID(UUID&& other) noexcept = default;
//     UUID& operator=(UUID&& other) noexcept = default;
//     virtual ~UUID() = default;

//     UUIDv4::UUID gen();

//     UUIDv4::UUID gen(const std::string& str);

//     std::string str();

//     static UUIDv4::UUID str_to_uuid(const std::string& str);

//     friend bool operator==(const UUID &lhs, const UUID &rhs) { return lhs.last_uuid == rhs.last_uuid; }
//     friend bool operator<(const UUID &lhs, const UUID &rhs) { return lhs.last_uuid < rhs.last_uuid; }
//     friend bool operator!=(const UUID &lhs, const UUID &rhs) { return !(lhs == rhs); }
//     friend bool operator> (const UUID &lhs, const UUID &rhs) { return rhs < lhs; }
//     friend bool operator<=(const UUID &lhs, const UUID &rhs) { return !(lhs > rhs); }
//     friend bool operator>=(const UUID &lhs, const UUID &rhs) { return !(lhs < rhs); }
// public:
//     UUIDv4::UUID last_uuid;
//     static UUIDv4::UUIDGenerator<std::mt19937_64> uuid_generator;
// };

class UUID
{
public:
    LAB_DEFAULT_CONSTRUCT(UUID);

    std::string gen_uuid();
    std::string gen_uuid(const std::string& str);
    uuids::uuid uuid() const;
    std::string str() const;
    static std::string gen();
    static std::string gen(const std::string& str);
    friend bool operator==(const UUID &lhs, const UUID &rhs) { return lhs.last_uuid == rhs.last_uuid; }
    friend bool operator!=(const UUID &lhs, const UUID &rhs) { return !(lhs == rhs); }
private:
    uuids::uuid last_uuid;
    std::string last_uuid_str;
public:
    static uuids::uuid_system_generator uuid_generator;
    static uuids::uuid_name_generator name_generator;
};

}

}