#ifndef _COMMON_H_
#define _COMMON_H_

#include <string>
#include <vector>
#include <utility>

struct StObject
{
    //marker info
    std::string camp;
    std::string targetName;
    double latitude;
    double longitude;
    std::string targetDirection;
    double targetSpeed;
    uint16_t targetState;
    uint8_t isWeapon;
    uint16_t hitRadius;
    std::string targetType;

    //box
    int x,y,w,h;
    float prob;
    unsigned int clsId;
    unsigned int objId;
    unsigned int age;
    double x3d,y3d,z3d;
    int lostframe;

    StObject():camp(""),targetName(""),latitude(0),longitude(0),targetDirection(""),targetSpeed(0),targetState(0),
    isWeapon(0),hitRadius(0),targetType(""),x(0),y(0),w(0),h(0),prob(0),clsId(0),objId(0),age(0),x3d(0),y3d(0),z3d(0){}

};

enum EnTaskCls
{
    EN_TS_NONE = 0,
    EN_TS_FUS = 1,
    EN_TS_ANA = 2
};


struct StMarkerDetailInfo
{
    std::string camp;
    std::string targetName;
    double latitude;
    double longitude;
    int targetCount;
    std::string targetDirection;
    double targetSpeed;
    int targetState;
    uint32_t isWeapon;
    uint32_t hitRadius;
    std::string targetType;
};

struct StMapMarkerInfo
{
    std::string timestampAndUserId;
    double latitude;
    double longitude;
    std::string markerUrl;
    StMarkerDetailInfo detailInfo;
    uint32_t jbMarkerCode;
    std::string jbColor;
    std::string setOption;
    uint64_t addMarkerTime;
    uint64_t delMarkerTime;
    uint64_t updateMarkerTime;
    std::string publisherUserId;
    std::string elevation;
    std::string intent;
    std::string grade;

    std::vector<std::pair<double, double>> pos;
};

struct StTsFusInputData
{
    std::string source;
    int i32Type;
    StMapMarkerInfo MapMarker;
};

struct StTsFusInput
{
    std::string sType;
    std::string sUserid;
    std::vector<StTsFusInputData> data;
};

struct StTsFusResultData
{
    std::string timestampAndUserId;
    StMapMarkerInfo markerInfo;
    int i32Type;
    int i32Res;
    std::string aiSource; //"zx"
};

struct StTsFusResultOutput
{
    int i32Type; //is 0
    std::vector<StTsFusResultData> data;
    std::string api;//"mapPlugin/ai"
};

//TS Analysis
struct StTsAnaData
{
    std::string source;
    int i32Type;
    int i32Time;
    StMapMarkerInfo  MapMarker;
};

struct StTsAnaInput
{
    std::string sType;
    std::string sUserid;
    StTsAnaData data;
};


struct StTsAnaResultData
{
    int i32Type;
    double latitude;
    double longitude;
    double angle;
    std::vector<StMapMarkerInfo> line;
    std::vector<StMapMarkerInfo> intent;
    std::vector<StMapMarkerInfo> shape;
};

struct StTsAnaResultOutput
{
    int i32Type; //is 1
    std::vector<StTsAnaResultData> data;
    std::string api;//"mapPlugin/ai"
};

//Intelligence Fusion
struct StIntelFusResultObj
{
    int id;
    std::string mapType;
    std::string camp;
    bool isWeapon;
    std::string lon_lat;
    std::string obj_bbox;
    std::string sTime;
    std::string sThreatDegree;
    std::string trend;
};

struct StIntelFusResultData
{
    std::string text;
    std::vector<StIntelFusResultObj> objs;
    std::string api;//"mapPlugin/ai"
};

struct StIntelFusResultOutput
{
    int i32Type; //is 2
    StIntelFusResultData data;
    std::string api;//"mapPlugin/ai"
};


//threat Analysis
struct StThreatResultData
{
    int id;
    std::string targetName;
    std::string grade;
};

struct StThreatResultOutput
{
    int i32Type; //is 3
    std::vector<StThreatResultData> data;
    std::string api;//"mapPlugin/ai"
};

#endif
