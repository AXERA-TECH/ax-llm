#pragma once
#include <iostream>
#include <string>
#include <map>

typedef void *(*create_fun)();
typedef void (*delete_fun)(void *);

class OBJFactory
{
public:
    ~OBJFactory() {}

    void *getObjectByID(int clsid, delete_fun &destroy_fun)
    {
        std::map<int, create_fun>::iterator it = m_id_map.find(clsid);
        if (it == m_id_map.end())
        {
            return NULL;
        }

        create_fun fun = it->second;
        if (!fun)
        {
            fprintf(stderr, "[%d] doesn`t REGISTER\n", clsid);
            return NULL;
        }
        void *obj = fun();
        destroy_fun = m_destroy_map[clsid];
        return obj;
    }

    bool contain(int clsid)
    {
        std::map<int, create_fun>::iterator it = m_id_map.find(clsid);
        if (it == m_id_map.end())
        {
            return false;
        }
        return true;
    }

    void registClass(int clsid, std::string name, create_fun fun, delete_fun destroy_fun)
    {
        m_id_map[clsid] = fun;
        m_destroy_map[clsid] = destroy_fun;
    }

    //??????
    static OBJFactory &getInstance()
    {
        static OBJFactory fac;
        return fac;
    }

private:
    OBJFactory() {}
    std::map<int, create_fun> m_id_map;
    std::map<int, delete_fun> m_destroy_map;
};

class RegisterAction
{
public:
    RegisterAction(int clsid, std::string className, create_fun ptrCreateFn, delete_fun ptrDestroyFn)
    {
        OBJFactory::getInstance().registClass(clsid, className, ptrCreateFn, ptrDestroyFn);
    }
};

#define REGISTER(clsid, className)                       \
    static void *objectCreator_##className()             \
    {                                                    \
        return new className;                            \
    }                                                    \
    static void objectDestroyer_##className(void *obj)   \
    {                                                    \
        delete (className *)obj;                         \
    }                                                    \
    static RegisterAction g_creatorRegister_##className( \
        clsid, #clsid, (create_fun)objectCreator_##className, (delete_fun)objectDestroyer_##className);