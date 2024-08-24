#include "renderer/utils/uuid.h"

namespace lab
{

namespace renderer
{

uuids::uuid_system_generator UUID::uuid_generator {};
uuids::uuid_name_generator UUID::name_generator{ uuid_generator() };

std::string UUID::gen_uuid()
{ 
    last_uuid = uuid_generator(); 
    last_uuid_str = uuids::to_string(last_uuid);
    return last_uuid_str;
}

std::string UUID::gen_uuid(const std::string& str)
{ 
    last_uuid = name_generator(str);
    last_uuid_str = uuids::to_string(last_uuid);
    return last_uuid_str;
}

uuids::uuid UUID::uuid() const
{
    return last_uuid;
}

std::string UUID::str() const
{
    return last_uuid_str;
}

std::string UUID::gen()
{
    return uuids::to_string(uuid_generator());
}

std::string UUID::gen(const std::string& str)
{
    return uuids::to_string(name_generator(str));
}

}

}