WA(2000)
'=====自动程序=====
DIM b AS INTEGER
VR(108)=VR(100)
WHILE TRUE
        IF TABLE(753)=2 THEN
            VR(1003)=2
        ENDIF
        IF TABLE(753)=3 THEN
            VR(1003)=3
        ENDIF
        IF TABLE(753)=1 THEN
            VR(1061)=INT(TABLE(752)*10000)/10000
            VR(1058)=INT(TABLE(750)*10000)/10000
            VR(1059)=INT(TABLE(751)*10000)/10000
            VR(1062)=VR(1061)+VR(1058)+VR(1059)
            TABLE(753,0)
            VR(1003)=1
        ENDIF
    '=====拍照时间=====
    IF rise(READ_OP(76),301) THEN
        b=TICKS
    ENDIF
    IF rise(VR(1003)=1,302) THEN
        VR(22)=2*ABS((TICKS-b)/1000)
        b=0
    ENDIF
    IF rise((IN(16) OR IN(112)),303) THEN'启动按钮
        IF VR(1).14=0 THEN
            IF VR(0).00=0 AND TABLE(100)=0 AND VR(0).20=1 THEN
                VR(0).00=1
                TABLE(100,1)
            ENDIF
            IF VR(0).00=1 AND TABLE(100)=0 THEN
                VR(0).00=0
                TABLE(100,1)
            ENDIF
        ELSE
            VR(0).04=1
            TABLE(100,1)
        ENDIF
    ENDIF
    IF rise((IN(19) OR IN(126)),304) THEN'暂停按钮
        IF VR(0).01=0 AND TABLE(100)=0 THEN
            VR(0).01=1
            TABLE(100,1)
        ENDIF
        IF VR(0).01=1 AND TABLE(100)=0 AND VR(0).20=1 THEN
            IF VR(82).14=0 THEN
                VR(0).01=0
                VR(0).04=1
            ELSE
                VR(0).08=1
                VR(0).09=1
            ENDIF
            TABLE(100,1)
        ENDIF
    ENDIF
    IF IN(21)=1 THEN
        IF delay(2,500) THEN'用50delay
            VR(83).16=1'气压不足的条件地址
            VR(0).01=1'暂停
            VR(2).06=0'前输送带连续
        ENDIF
    ENDIF

    '=====自动运行=====
    IF fall(VR(0).01,314) AND (VR(0).00=1 OR VR(28)>0) THEN
        VR(1).13=1
        IF VR(0).00=1 THEN
            VR(1).16=1
        ENDIF
    ENDIF
    IF IN(24)=0 AND VR(0).00=1 THEN
        VR(80).16=1
        VR(0).01=1
    ENDIF
    IF (rise(VR(0).00,305) AND VR(1).14=0) OR VR(1).13=1 THEN
        VR(2).14=0
        VR(2).16=0
        IF VR(2).00=0 THEN
            VR(83).12=1
            VR(0).00=0
        ENDIF
        IF IN(24)=0 THEN
            VR(80).16=1
            VR(0).00=0
            VR(0).01=1
        ENDIF
        IF VR(1005)=0 THEN'触发模式***开
            VR(80).11=1
            VR(0).00=0
        ENDIF
        IF VR(1004)=0 THEN'视觉离线报警
            VR(83).05=1
            VR(0).00=0
        ENDIF
        IF VR(5)=0 THEN
            VR(80).01=1
            VR(0).00=0
        ENDIF
        IF IN(39)=1 THEN
            VR(82).13=1
            VR(0).00=0
        ENDIF
        IF VR(0).05=0 THEN
            VR(82).08=1
            VR(0).00=0
        ENDIF
        IF VR(61)>2 AND VR(61)<358 AND MTYPE AXIS(1)=0 THEN
            VR(80).02=1
            VR(0).00=0
        ENDIF
        IF ABS(VR(1058))<1 OR ABS(VR(1059))<1 THEN
            VR(1058)=VR(1054)
            VR(1059)=VR(1055)
            IF ABS(VR(1054))<1 OR ABS(VR(1055))<1 THEN
                VR(83).01=1
                VR(0).00=0
            ENDIF
        ENDIF
        IF VR(1).14=1 THEN
            VR(0).00=0
            IF VR(83).14=0 THEN
                VR(0).01=1
            ENDIF
        ENDIF
        IF VR(5)=1 AND VR(0).05=1 AND IN(39)=0 AND VR(83).00=0 AND VR(80).11=0 AND VR(80).16=0_
                AND VR(1004)=1 AND VR(1005)=1 AND VR(82).09=0 AND VR(80).02=0 AND VR(1).14=0 AND VR(83).01=0 THEN
            VR(3).00=0
            VR(3).01=0
            VR(3).02=0
            VR(3).03=0
            VR(3).04=0
            VR(3).05=0
            VR(3).06=0
            VR(3).07=0
            VR(3).08=0
            VR(3).09=0
            TABLE(102,0)
            VR(2).02=1'真空泵
            VR(2).04=1'吸风
            VR(2).06=0
            VR(25)=0
            IF VR(0).00=1 THEN
                VR(28)=0
            ENDIF
            VR(50)=0
        ENDIF
    ENDIF
    IF VR(5)=1 AND VR(0).05=1 AND IN(39)=0 AND VR(83).00=0 AND VR(80).11=0 AND VR(1004)=1 AND VR(1005)=1 _
            AND VR(82).09=0 AND VR(0).01=0 AND (VR(0).00=1 OR VR(28)>0) AND VR(29)=0 THEN
        IF READ_OP(72)=0 THEN
            IF delay(3,300) THEN
                OP(72,1)
            ENDIF
        ENDIF
        IF delay(4,3000) OR VR(1).16=1 THEN
            OP(72,0)
            VR(29)=1
            VR(1).13=0
            VR(1).16=0
            IF VR(0).00=1 THEN
                IF IN(36)=0 THEN
                    VR(30)=VR(30)+1
                ENDIF
                IF VR(21).11=0 AND ABS(MPOS AXIS(3))<1 THEN
                    VR(30)=VR(30)+1
                ENDIF
                VR(3).16=0
            ENDIF
        ENDIF
    ENDIF
    IF VR(0).01 THEN
        VR(29)=0
    ENDIF
    IF VR(0).00=0 THEN
        VR(1).13=0
        VR(1).16=0
    ENDIF

    '=====自动停止=====
    IF VR(0).00=0 AND VR(29)=1 THEN
        IF VR(28)=0 THEN
            VR(28)=1'停机计数
        ENDIF
        IF VR(28)>=VR(115) THEN
            VR(29)=0
            VR(28)=0
        ENDIF
    ENDIF
    '=====飞达卡纸停止=====
    IF VR(25)>2 AND VR(0).00=1 THEN
        VR(0).00=0
        VR(80).10=1
        VR(25)=0
    ENDIF


'=====产量到达停止=====
    IF VR(26)>= VR(101) AND VR(101)>0 AND VR(29)=1 THEN
        VR(26)=0
        VR(0).00=0
    ENDIF
    '=====飞达单步=====
    IF (IN(25)=1 OR IN(99)=1) AND TABLE(100)=0 AND VR(0).20=1 THEN
        IF IN(24)=1 THEN
            VR(0).06=1
            TABLE(100,1)
        ELSE
            VR(80).16=1
        ENDIF
    ENDIF

    IF rise(VR(0).06,306) THEN
        IF (VR(29)=0 OR (VR(29)=1 AND VR(0).01)) THEN
            VR(2).14=0
            VR(2).16=0
            VR(1003)=0
            TABLE(753,0)
            IF VR(2).00=0 THEN
                VR(83).12=1
                VR(0).06=0
            ENDIF
            IF IN(21)=1 THEN
                VR(83).16=1
                VR(0).06=0
            ENDIF
            IF IN(24)=0 THEN
                VR(80).16=1
                VR(0).06=0
            ENDIF
            IF VR(1005)=0 THEN'触发模式未开
                VR(80).11=1
                VR(0).06=0
            ENDIF
            IF VR(1004)=0 THEN'视觉离线报警
                VR(83).05=1
                VR(0).06=0
            ENDIF
            IF VR(2).02=0 OR VR(2).04=0 THEN '真空泵或吸风没有打开
                VR(0).06=0
                VR(83).02=1
            ENDIF
            IF VR(5)=0 THEN
                VR(80).01=1
                VR(0).06=0
            ENDIF
            IF VR(0).05=0 THEN
                VR(82).08=1
                VR(0).06=0
            ENDIF
            IF IN(39)=1 THEN
                VR(82).13=1
                VR(0).06=0
            ENDIF
            IF VR(5)=1 AND IN(39)=0 AND VR(0).05=1 AND VR(2).02=1 AND VR(2).04=1_
                AND VR(1004)=1 AND VR(1005)=1 AND VR(0).06=1 THEN'真空泵吸风同时打开
                VR(20).00=1
                VR(3).21=1
            ENDIF
        ELSE
            VR(0).06=0
        ENDIF
    ENDIF
    '=====输送单步=====
    IF (IN(23)=1 OR IN(125)=1) AND TABLE(100)=0 AND MTYPE AXIS(40)=0 AND VR(0).20=1 THEN
        IF IN(24)=1 THEN
            VR(0).07=1
            TABLE(100,1)
        ELSE
            VR(80).16=1
        ENDIF
    ENDIF
    IF rise(VR(0).07,307) THEN
        IF (VR(29)=0 OR (VR(29)=1 AND VR(0).01)) THEN
            IF MTYPE AXIS(40)=1 OR MTYPE AXIS(40)=5 THEN
                VR(0).07=0
            ENDIF
            IF IN(21)=1 THEN
                VR(83).16=1
                VR(0).07=0
            ENDIF
            IF IN(24)=0 THEN
                VR(80).16=1
                VR(0).07=0
            ENDIF
            IF VR(0).07=1 THEN
                VR(2).14=0
                VR(2).16=0
                VR(1003)=0
                TABLE(753,0)
                IF VR(1005)=0 THEN'触发模式未开
                    VR(80).11=1
                    VR(0).07=0
                ENDIF
                IF VR(1004)=0 THEN'视觉离线报警
                    VR(83).05=1
                    VR(0).07=0
                ENDIF
                IF VR(2).02=0 OR VR(2).04=0 THEN'真空泵或者吸风都关闭
                    VR(0).07=0
                    VR(83).02=1
                ENDIF
                IF VR(5)=0 THEN
                    VR(80).01=1
                    VR(0).07=0
                ENDIF
                IF VR(0).05=0 THEN
                    VR(82).08=1
                    VR(0).07=0
                ENDIF
                IF VR(5)=1 AND VR(0).05=1 AND VR(2).02=1 AND VR(2).04=1_
                    AND VR(1004)=1 AND VR(1005)=1 THEN'真空泵或吸风
                    VR(20).01=1
                ENDIF
            ENDIF
        ELSE
            VR(0).07=0
        ENDIF
    ENDIF
'=====机械手单步=====
    IF rise(VR(0).08,308) AND VR(0).20=1 THEN
        IF (VR(29)=0 OR (VR(29)=1 AND VR(0).01)) THEN
            VR(2).14=0
            VR(2).16=0
            VR(1003)=0
            TABLE(753,0)
            IF VR(1005)=0 THEN'触发模式未开
                VR(80).11=1
                VR(0).08=0
            ENDIF
            IF IN(21)=1 THEN
                VR(83).16=1
                VR(0).08=0
            ENDIF
            IF VR(1004)=0 THEN'视觉离线报警
                VR(83).05=1
                VR(0).08=0
            ENDIF
            IF VR(2).02=0 OR VR(2).04=0 THEN'真空泵或者吸风
                VR(0).08=0
                VR(83).02=1
            ENDIF
            IF VR(5)=0 THEN
                VR(80).01=1
                VR(0).08=0
            ENDIF
            IF VR(0).05=0 THEN
                VR(82).08=1
                VR(0).08=0
            ENDIF
            IF ABS(VR(1058))<1 OR ABS(VR(1059))<1 THEN
                VR(1058)=VR(1054)
                VR(1059)=VR(1055)
                IF ABS(VR(1054))<1 OR ABS(VR(1055))<1 THEN
                    VR(83).01=1
                    VR(0).08=0
                ENDIF
            ENDIF
            IF VR(5)=1 AND VR(0).05=1 AND VR(2).02=1 AND VR(2).04=1_
                AND VR(1004)=1 AND VR(1005)=1 AND VR(0).08=1 THEN'真空泵吸风都打开
                VR(20).02=1
            ENDIF
        ELSE
            VR(0).08=0
        ENDIF
    ENDIF

    '=====面纸双张=====
    IF IN(41)=0 AND VR(21).14=0 THEN
        VR(83).14=1'面纸双张
        IF VR(0).00=1 THEN
            VR(0).00=0
        ENDIF
    ENDIF
'=====飞达工艺=====
    IF VR(21).07=0 THEN
        VR(331)=3
        VR(330)=0
    ENDIF
    IF rise(VR(21).07,309) THEN
        VR(331)=0
    ENDIF
    IF VR(331)>100 THEN
        VR(331)=100
    ENDIF
    IF VR(331)<3 THEN
        VR(331)=3
    ENDIF
    VR(332)=VR(331)+VR(330)

    IF VR(332)>100 THEN
        VR(332)=100
    ENDIF
    IF VR(332)<3 THEN
        VR(332)=3
    ENDIF


    IF VR(21).07=1 THEN
        VR(134)=VR(144)
        VR(327)=VR(134)-VR(103)+50
    ELSE
        VR(134)=VR(123)
    ENDIF

    IF VR(312)>450 THEN
        VR(312)=450
    ENDIF
    IF VR(312)<1 THEN
        VR(312)=1
    ENDIF
    VR(314)=VR(312)+VR(313)
    IF VR(314)>450 THEN
        VR(314)=450
    ENDIF
    IF VR(314)<1 THEN
        VR(314)=1
    ENDIF
    IF VR(0).00=0 AND VR(29)=0 AND VR(21).13=0 THEN
        VR(30)=0
    ENDIF

    IF IN(37)=0 THEN
        IF delay(5,1000) THEN
            VR(4).20=0
        ENDIF
    ELSE
        IF delay(6,2000) THEN
            VR(4).20=1
            VR(79).00=0
        ENDIF
    ENDIF
    IF (IN(37)=1 OR VR(5022).02=1 OR VR(2).17=1) AND IN(127)=0 THEN
        VR(2).18=1
    ELSE
        VR(2).18=0
    ENDIF
    '=====导纸工艺=====
    SELECT_CASE VR(6)
        CASE 0
            IF VR(20).00=1 OR (VR(0).00=1 AND VR(29)=1 AND VR(28)=0 AND VR(0).01=0 AND VR(2).18=1) AND VR(7)=0 AND _
                    MTYPE AXIS(6)=0 AND MTYPE AXIS(2)=0 AND MTYPE AXIS(1)=0 AND VR(83).18=0 THEN
                IF VR(102)=VR(100) AND VR(3).18=0 THEN
                    IF VR(20).00=0 AND VR(4).17=1 THEN
                        VR(83).18=1'缺铁片
                        VR(0).01=1
                        VR(4).17=0
                    ELSE
                        OFFPOS AXIS(6)=-DPOS AXIS(6)
                        OFFPOS AXIS(2)=-DPOS AXIS(2)
                        WAIT UNTIL DPOS AXIS(2)<0.1
                        IF delay(7,VR(143)*1000) THEN
                            IF VR(20).00=0 AND IN(108)=1 THEN
                                VR(80).18=1'翻转机暂停
                            ENDIF
                            IF VR(20).00=0 AND VR(4).20=0 THEN
                                VR(79).00=1'灰板存料下限
                            ENDIF
                            IF VR(20).00=0 AND VR(4).17=1 THEN
                                VR(83).18=1'缺纸舌或铁片
                                VR(0).01=1
                                VR(4).17=0
                            ENDIF
                            IF VR(80).18=0 AND VR(83).18=0 AND VR(79).00=0 THEN
                                IF (VR(30)>0 OR VR(20).00=1) AND VR(20).01=0 THEN
                                    IF DPOS AXIS(1)>359 OR DPOS AXIS(1)<1 THEN'在飞达原点上
                                        MOVELINK(0,1,0,0,6,2,VR(314))AXIS(2)
                                        MOVE(VR(52)) AXIS(2)
                                        MOVE(VR(134)) AXIS(6)
                                        MOVELINK(80,160,50,0,6,2+16,VR(332))AXIS(1)
                                        MOVE(280) AXIS(1)
                                        OPEN_WIN AXIS(6)=(VR(103)-50)
                                        CLOSE_WIN AXIS(6)=VR(134)
                                        REGIST(20,20,4,0,1)AXIS(6)
                                        VR(7)=1
                                        VR(6)=2
                                        IF VR(0).00=1 AND VR(0).01=0 AND IN(20)=1 THEN
                                            VR(30)=VR(30)-1
                                        ENDIF

                                        VR(157)=VR(156)
                                        VR(156)=TICKS
                                        IF VR(157)<>0 AND ABS(VR(157)-VR(156))>1 THEN
                                            VR(158)=30000/(VR(157)-VR(156))
                                        ENDIF
                                        IF IN(20)=1 AND VR(0).00=1 THEN
                                            VR(3).00=1'导纸启动时飞达有***
                                        ENDIF
                                    ELSE
                                        VR(6)=1
                                    ENDIF
                                ENDIF
                            ENDIF
                        ENDIF
                    ENDIF
                ENDIF
            ENDIF
        CASE 1'不达飞达原点上
            IF DPOS AXIS(1)>0.5 AND DPOS AXIS(1)<=180 AND MTYPE AXIS(1)<>2 THEN
                CANCEL(2) AXIS(1)
                MOVEABS(0) AXIS(1)
            ENDIF
            IF DPOS AXIS(1)>180 AND DPOS AXIS(1)<=359.5 AND MTYPE AXIS(1)<>2 THEN
                CANCEL(2) AXIS(1)
                MOVEABS(360) AXIS(1)
            ENDIF
            IF DPOS AXIS(1)>359 OR DPOS AXIS(1)<1 AND MTYPE AXIS(1)=0 THEN
                VR(6)=0
            ENDIF
        CASE 2 '导纸启动
            IF MARK AXIS(6)=0 THEN
                VR(20).00=0
                VR(6)=3
            ELSE
                IF delay(8,200) AND IN(20)=0 THEN
                    REGIST(20,20,4,0,0)AXIS(6)
                ENDIF
            ENDIF
        CASE 3
            IF MARK AXIS(6)=-1 AND DPOS AXIS(6)>REG_POS AXIS(6)+1 THEN
                IF IN(20)=1 THEN'正常信号
                    VR(326)=REG_POS AXIS(6)
                    VR(450)=VR(327)-VR(326)
                    MOVEMODIFY(VR(326)+50)AXIS(6)
                    VR(25)=0
                    IF VR(21).07=1 THEN'导纸排两张
                        IF VR(21).12=0 AND VR(6)=3 THEN
                            VR(21).12=1'第一张
                            VR(330)=0
                            VR(331)=VR(331)+VR(450)
                            VR(6)=4
                        ENDIF
                        IF VR(21).12=1 AND VR(6)=3 THEN
                            VR(21).12=0
                            VR(330)=VR(450)
                            VR(6)=4
                        ENDIF
                    ENDIF
                    IF VR(21).07=0 THEN
                        VR(6)=4
                        VR(21).12=0
                    ENDIF
                ENDIF
                IF IN(20)=0 THEN'干扰信号
                    REGIST(20,20,4,0,0)AXIS(6)
                ENDIF
            ENDIF
            IF VR(6)=3 AND ABS(DPOS AXIS(6)-VR(134))<1 AND MARK AXIS(6)=0 THEN
                VR(6)=4
                VR(330)=0
                VR(450)=0
                VR(21).12=0
                IF VR(0).00=1 THEN
                    VR(25)=VR(25)+1
                ENDIF
                IF IN(20)=1 THEN
                    VR(80).14=1
                    VR(0).00=0
                ENDIF
            ENDIF
        CASE 4'等待结束
            IF VR(7)=0 THEN
                VR(0).06=0
                VR(6)=0
            ENDIF
    END_CASE

    IF VR(21).13=1 THEN
        IF VR(3).15=1 AND VR(30)<2 THEN
            VR(30)=2
        ENDIF
        IF VR(3).15=0 THEN
            VR(30)=0
        ENDIF
    ENDIF
    IF IN(38) AND VR(0).00 THEN
        VR(83).17=1
        VR(0).01=1
    ENDIF
    '=====输送工艺=====
    IF (VR(54)=1 AND VR(3).02=1) OR (VR(54)=2 AND VR(3).03=1) OR (VR(54)=3 AND VR(3).04=1)_
            OR (VR(54)=4 AND VR(3).05=1) OR (VR(54)=5 AND VR(3).06=1) OR (VR(54)=6 AND VR(3).07=1)_
            OR (VR(54)=7 AND VR(3).08=1) OR (VR(54)=8 AND VR(3).09=1) THEN
        VR(0).15=1'胶辊卡纸用
    ELSE
        VR(0).15=0
    ENDIF
    IF VR(180)<=0.1 OR VR(180)>50 THEN
        VR(180)=1.5
    ENDIF
    IF (VR(28)>0 OR VR(2).18=0) AND VR(29)=1 AND VR(7)=0 AND VR(20).01=0 AND VR(0).01=0 _
            AND (VR(8)=0 OR VR(8)=6 OR VR(8)=14) THEN
        IF delay(9,(VR(180)+VR(143)+1)*1000) THEN
            VR(20).01=1
        ENDIF
    ENDIF

    IF IN(109)=1 OR VR(3).13=1 AND VR(29)=1 THEN
        IF VR(3).14=0 OR VR(0).00=1 THEN
            VR(3).14=1
            VR(28)=1
        ENDIF
    ELSE
        VR(3).14=0
    ENDIF

    IF fall(VR(3).14,145) AND VR(29)=1 AND VR(0).00=1 THEN
        VR(28)=0
    ENDIF

    IF VR(7)=22 OR VR(7)=23 THEN
        IF DPOS AXIS(2)>=10 AND pad=0 THEN
            VR(3).09=0
            OP(55,1)
            IF VR(0).15=1 AND VR(21).15=1 THEN'铁片检测
                VR(4).17=1
            ENDIF
            IF VR(3).08=1 THEN
                VR(3).09=1
                VR(3).08=0
            ENDIF
            IF VR(3).07=1 THEN
                VR(3).08=1
                VR(3).07=0
            ENDIF
            IF VR(3).06=1 THEN
                VR(3).07=1
                VR(3).06=0
            ENDIF
            IF VR(3).05=1 THEN
                VR(3).06=1
                VR(3).05=0
            ENDIF
            IF VR(3).04=1 THEN
                VR(3).05=1
                VR(3).04=0
            ENDIF
            IF VR(3).03=1 THEN
                VR(3).04=1
                VR(3).03=0
            ENDIF
            IF VR(3).02=1 THEN
                VR(3).03=1
                VR(3).02=0
            ENDIF
            IF VR(3).01=1 THEN
                VR(3).02=1
                VR(3).01=0
            ENDIF
            IF VR(3).00=1 THEN
                VR(3).01=1
                VR(3).00=0
            ENDIF
            pad=1
        ENDIF
        IF ABS(DPOS AXIS(2)-VR(52))<0.5 AND MTYPE AXIS(2)=0 AND MARK AXIS(2)=0 THEN
            IF VR(0).15=1 AND VR(0).00=1 AND VR(0).01=0 THEN
                VR(80).13=1
                VR(0).01=1
            ENDIF
            VR(311)=0
            VR(313)=0
            youzhi1=VR(180)*1000
            VR(7)=8
            OP(65,1)
            OP(55,0)
        ENDIF
    ENDIF
    SELECT_CASE VR(7)
        CASE 0'单步
            IF IN(22)=0 THEN
                luibo=0
            ENDIF
            IF VR(20).01=1 AND VR(7)=0 AND MTYPE AXIS(2)=0 AND MTYPE AXIS(40)=0 AND VR(83).18=0 THEN
                OFFPOS AXIS(2)=-DPOS AXIS(2)
                WAIT UNTIL DPOS AXIS(2)<0.5
                IF VR(20).00=0 AND VR(4).17=1 THEN
                    VR(4).17=0
                ELSE
                    MOVE(VR(52)) AXIS(2)
                    VR(7)=1
                ENDIF
            ENDIF
        CASE 1
            IF IN(22)=0 THEN
                REGIST(20,22,4,0,0)AXIS(2)
                VR(7)=21
                IF VR(28)>0 THEN
                    VR(28)=VR(28)+1
                ENDIF
                pad=0
            ELSE
                IF luibo=1 THEN
                    luiz=VR(103)-(VR(441)-REG_POS AXIS(2))
                    IF luiz>VR(120) THEN
                        OPEN_WIN AXIS(2)=luiz-VR(120)
                        CLOSE_WIN AXIS(2)=VR(52)
                        REGIST(20,22,4,1,1)AXIS(2)
                        IF VR(28)>0 THEN
                            VR(28)=VR(28)+1
                        ENDIF
                        pad=0
                        VR(7)=3
                    ELSE
                        VR(7)=2
                    ENDIF
                ELSE
                    VR(7)=2
                ENDIF
            ENDIF
        CASE 21'开始时没纸
            IF MARK AXIS(2)=0 THEN
                VR(7)=22
            ENDIF
        CASE 22'上升沿后，滤波检测下降沿
            IF MARK AXIS(2) AND DPOS AXIS(2)>REG_POS AXIS(2)+1 THEN
                IF IN(22)=1 THEN
                    VR(441)=VR(52)
                    OPEN_WIN AXIS(2)=REG_POS AXIS(2)+VR(103)-VR(120)
                    CLOSE_WIN AXIS(2)=VR(52)
                    REGIST(20,22,4,1,1)AXIS(2)
                    VR(7)=23
                ELSE
                    REGIST(20,22,4,0,0)AXIS(2)
                    VR(7)=21
                ENDIF
            ENDIF
        CASE 23'下降沿启动成功
            IF MARK AXIS(2)=0 THEN
                VR(7)=4
            ENDIF


        CASE 2'启动色标捕捉
            REGIST(20,22,4,1,0)AXIS(2)
            IF VR(28)>0 THEN
                VR(28)=VR(28)+1
            ENDIF
            pad=0
            VR(7)=3
        CASE 3
            IF MARK AXIS(2)=0 THEN
                VR(7)=4
                luibo=0
            ENDIF
        CASE 4'色标启用成功
            IF DPOS AXIS(2)>=10 AND pad=0 THEN
                VR(3).09=0
                OP(55,1)
                IF VR(0).15=1 AND VR(21).15=1 THEN'铁片检测
                    VR(4).17=1
                ENDIF
                IF VR(3).08=1 THEN
                    VR(3).09=1
                    VR(3).08=0
                ENDIF
                IF VR(3).07=1 THEN
                    VR(3).08=1
                    VR(3).07=0
                ENDIF
                IF VR(3).06=1 THEN
                    VR(3).07=1
                    VR(3).06=0
                ENDIF
                IF VR(3).05=1 THEN
                    VR(3).06=1
                    VR(3).05=0
                ENDIF
                IF VR(3).04=1 THEN
                    VR(3).05=1
                    VR(3).04=0
                ENDIF
                IF VR(3).03=1 THEN
                    VR(3).04=1
                    VR(3).03=0
                ENDIF
                IF VR(3).02=1 THEN
                    VR(3).03=1
                    VR(3).02=0
                ENDIF
                IF VR(3).01=1 THEN
                    VR(3).02=1
                    VR(3).01=0
                ENDIF
                IF VR(3).00=1 THEN
                    VR(3).01=1
                    VR(3).00=0
                ENDIF
                pad=1
            ENDIF
            IF ABS(DPOS AXIS(2)-VR(52))<0.5 AND MTYPE AXIS(2)=0 AND MARK AXIS(2)=0 THEN
                IF VR(0).15=1 AND VR(0).00=1 AND VR(0).01=0 THEN
                    VR(80).13=1
                    VR(0).01=1
                ENDIF
                VR(311)=0
                VR(313)=0
                youzhi1=VR(180)*1000
                VR(7)=8
                OP(65,1)
            ENDIF

            IF MARK AXIS(2) AND DPOS AXIS(2)>REG_POS AXIS(2)+1 THEN
                IF IN(22)=0 THEN
                    IF VR(29)=1 THEN
                        IF VR(101)>0 THEN'批产量
                            VR(26)=VR(26)+1
                        ENDIF
                        VR(24)=VR(24)+1
                        VR(23)=VR(23)+1
                        VR(85)=VR(85)+1
                        youzhi1=0
                    ENDIF
                    VR(307)=INT((REG_POS AXIS(2)-0.001*REGIST_SPEED AXIS(2))*1000)/1000
                    VR(441)=VR(307)+VR(55)
                    MOVEMODIFY(VR(441)) AXIS(2)
                    VR(310)=VR(307)-VR(308)'+vr(55)理论与实际差
                    IF VR(0).07=0 THEN
                        VR(311)=VR(311)+1'第几张
                        IF VR(311)>=VR(54)+1 THEN
                            VR(311)=0
                        ENDIF
                        IF VR(311)=1 THEN
                            VR(312)=VR(312)+VR(310)
                            VR(313)=0
                        ENDIF
                        IF VR(311)<>1 THEN
                            VR(313)=VR(310)
                        ENDIF
                    ENDIF
                    IF VR(0).07=1 THEN
                        VR(311)=0
                        VR(313)=0
                    ENDIF
                    REGIST(20,22,4,0,0)AXIS(2)
                    VR(7)=5
                ENDIF
                IF IN(22)=1 THEN
                    IF DPOS AXIS(2)>20 AND DPOS AXIS(2)<VR(52)-50 THEN
                        REGIST(20,22,4,1,0)AXIS(2)
                        VR(7)=3
                    ELSE
                        VR(80).06=1
                        VR(0).01=1
                        VR(7)=8
                    ENDIF
                ENDIF
            ENDIF
        CASE 5
            IF MARK AXIS(2)=0 THEN'下一张启动成功
                luibo=1
            ENDIF

            IF ABS(DPOS AXIS(2)-VR(441))<1 AND MTYPE AXIS(2)=0 THEN'到目标位置
                IF delay(10,VR(174)*1000) THEN
                    VR(7)=6
                    OP(65,1)
                    OP(55,0)
                    VR(2).15=1 '输送有纸
                ENDIF
            ENDIF
        CASE 6'等待对位
            IF VR(8)=5 AND VR(2).15=0 THEN
                VR(7)=8
            ENDIF

        CASE 7'已经放板
            IF VR(6)=0 OR VR(6)=4 THEN
                VR(7)=8
            ENDIF
        CASE 8
            VR(2).15=0
            VR(7)=0
            VR(0).07=0
            VR(20).01=0
    END_CASE
    IF READ_OP(65)=1 THEN
        IF delay(11,VR(145)*1000) THEN
            OP(65,0)
        ENDIF
    ENDIF
    '=====拍照=====
    IF VR(415)>VR(414) THEN
        PSWITCH(2,1,40,76,1,VR(414),VR(415))
    ENDIF
'=====机械手工艺=====
    IF VR(2).02=0 AND MTYPE AXIS(40)=0 THEN
        PSWITCH(1,1,40,59,1,1001,1002)
    ENDIF
    SELECT_CASE VR(8)
        '机器人抓取
        CASE 0
            IF (VR(7)=0 OR VR(7)=6 OR VR(29)=1) AND (VR(20).02=1 OR (VR(29)=1 AND VR(0).01=0) OR (VR(0).06=1 OR VR(0)._
                    07=1 AND VR(2).15=1)) AND MTYPE AXIS(40)=0 THEN
                OFFPOS AXIS(40)=-DPOS AXIS(40)
                IF DPOS AXIS(40)>999.99 OR DPOS AXIS(40)<0.01 AND VR(102)=VR(100) AND VR(3).18=0 THEN
                    VR(8)=1
                ENDIF
            ENDIF
        CASE 1
            IF NTYPE AXIS(20)=0 AND NTYPE AXIS(21)=0 AND MTYPE AXIS(40)=0 AND NTYPE AXIS(23)=0 THEN
                CANCEL(2) AXIS(22)
                MOVELINK(INT((VR(1050)-DPOS AXIS(20))*1000)/1000,VR(437),0.25*VR(437),0.25*VR(437),40,2+16,VR(416))_
                    AXIS(20)'回原点
                MOVELINK(INT((VR(1051)-DPOS AXIS(21))*1000)/1000,VR(437),0.25*VR(437),0.25*VR(437),40,2+16,VR(416))_
                    AXIS(21)
                MOVELINK((VR(1052)+VR(131)-DPOS AXIS(22)),VR(437),1,1,40,2+16,VR(416))AXIS(22)
                MOVELINK(INT((VR(1053)-DPOS AXIS(23))*10000)/10000,VR(437),1,1,40,2+16,VR(416))AXIS(23)
                IF VR(1003)=1 THEN
                    PSWITCH(0,1,40,60,1,VR(424),VR(425))
                    VR(1008)=VR(1008)+1
                ENDIF
                VR(8)=2
                IF VR(9)=0 OR VR(9)=9 AND IN(36)=1 THEN
                    VR(21).11=0
                ENDIF
            ENDIF
        CASE 2
            IF DPOS AXIS(40)<VR(416)+0.5 THEN
                VR(2).12=0
                PSWITCH(0,1,40,60,1,VR(424),VR(425))
                MOVEABS(VR(417)-5)AXIS(40)'走至取好板
                VR(8)=14
            ENDIF
        CASE 14
            IF VR(9)=2 OR IN(36)=0 OR VR(0).08 THEN
                MOVEABS(VR(420))AXIS(40)'走至取好板
                VR(3).17=0
                VR(8)=13
            ENDIF
        CASE 13
            IF VR(407)<80 THEN
                IF VR(411)>0.1 AND VR(411)<1.0 AND VR(0).00=0 THEN
                    VR(411)=VR(411)-0.05
                ENDIF
                VR(80).17=1
                VR(0).01=1
            ENDIF
            IF DPOS AXIS(40)<VR(420) AND DPOS AXIS(40)>VR(416) THEN
                VR(1003)=0
                TABLE(753,0)
                PSWITCH(1,1,40,59,0,1001,1002)
                PSWITCH(0,1,40,60,1,1001,1002)
                VR(0).19=0
                IF IN(36)=1 AND VR(8)=13 AND NTYPE AXIS(22)=0 THEN
                    MOVELINK(-VR(131),VR(405),0.2*VR(405),0.2*VR(405),40,16)AXIS(22)'原点下降
                    VR(8)=3
                    VR(3).17=1
                    VR(4).11=0
                ENDIF
                IF IN(36)=0 AND VR(8)=13 THEN
                    VR(8)=5
                ENDIF
            ENDIF
        CASE 3
            IF DPOS AXIS(40)>VR(416) THEN
                MOVELINK(VR(131),VR(405),0.2*VR(405),0.2*VR(405),40,16+2,VR(419))AXIS(22)'原点上升
                VR(8)=4
            ENDIF
        CASE 4
            IF DPOS AXIS(40)>VR(419) THEN
                VR(8)=5
            ENDIF
        CASE 5
            IF DPOS AXIS(40)>VR(416) AND DPOS AXIS(40)<VR(420) THEN
                VR(8)=6
            ENDIF
        CASE 6
            'SCARA去放置点位置
            IF (VR(7)=0 OR VR(7)=6) AND (VR(20).02=1 OR ((VR(29)=1 OR VR(0).06=1 OR VR(0).07=1) AND VR(2).15=1)) THEN
                MOVELINK(INT((VR(1054)-VR(1050))*1000)/1000,VR(407),0.25*VR(407),0.25*VR(407),40,2+16,VR(421))AXIS(20)
'去放置点
                MOVELINK(INT((VR(1055)-VR(1051))*1000)/1000,VR(407),0.25*VR(407),0.25*VR(407),40,2+16,VR(421))AXIS(21)
                MOVELINK(INT((VR(1057)-VR(1053))*10000)/10000,VR(407),0.25*VR(407),0.25*VR(407),40,2+16,VR(421))AXIS(23)
                VR(8)=17
            ENDIF
            IF DPOS AXIS(40)>=VR(419) AND VR(9)=2 AND VR(3).17=1 THEN
                VR(3).17=0
                VR(21).11=1
                VR(0).19=1
            ENDIF
            IF ABS(DPOS AXIS(40)-VR(420))<0.01 AND MTYPE AXIS(40)=0 THEN
                IF VR(0).00=0 AND VR(3).18=1 THEN
                    OFFPOS AXIS(40)=-DPOS AXIS(40)
                    VR(8)=0
                ENDIF
            ENDIF
        CASE 17
            IF VR(3).17=0 OR VR(9)=2 THEN
                VR(8)=7
                MOVEABS(1000) AXIS(40)
            ENDIF
        CASE 7
            IF NTYPE AXIS(20)=0 AND NTYPE AXIS(21)=0 AND NTYPE AXIS(22)=0 AND NTYPE AXIS(23)=0 THEN
                IF DPOS AXIS(40)>=VR(421) THEN
                    VR(8)=8
                    IF VR(9)=2 AND VR(3).17=1 THEN
                        VR(3).17=0
                        VR(21).11=1
                        VR(0).19=1
                    ENDIF
                    IF VR(0).19=1 AND IN(36)=0 THEN
                        VR(4).02=1
                    ENDIF
                ENDIF
            ENDIF
        CASE 8
            IF DPOS AXIS(40)>(VR(421)+VR(407)) THEN
                VR(4).11=1
                IF VR(8)=8 THEN
                    IF IN(36)=1 THEN
                        VR(0).01=1
                        VR(82).14=1
                    ENDIF
                    IF ABS(VR(1003)-2)<0.5 THEN
                        VR(0).01=1
                        VR(82).15=1
                    ENDIF
                ENDIF
                IF ABS(VR(1057)-VR(1061))>VR(177) THEN
                    VR(0).01=1
                    VR(83).01=1
                    VR(1).12=0
                ENDIF
                IF ABS(VR(1054)-VR(1058))>VR(179) THEN
                    VR(0).01=1
                    VR(83).01=1
                    VR(1).12=0
                ENDIF
                IF ABS(VR(1003)-1)>0.5 THEN
                    VR(0).01=1
                    VR(83).03=1
                ENDIF
                IF DPOS AXIS(40)>=VR(423) AND VR(4).16=1 THEN
                    VR(8)=0
                    PSWITCH(0,1,40,60,1,1001,1002)
                    VR(0).01=1
                    VR(20).02=0
                    VR(0).06=0
                    VR(0).07=0
                    VR(0).08=0
                    VR(83).03=1
                    VR(4).16=0
                ENDIF
                IF VR(0).19=1 AND IN(36)=0 THEN
                    VR(21).11=0
                ENDIF
                IF ( (VR(83).06=1) OR (VR(83).03=1) OR (IN(36)=1) OR (ABS(VR(1057)-VR(1061))>VR(177))) _
                        AND VR(8)=8 THEN
                    VR(0).08=0
                    VR(0).06=0
                    VR(0).07=0
                    VR(20).02=0
                    VR(8)=0
                    VR(0).01=1
                    PSWITCH(0,1,40,60,1,1001,1002)
                ENDIF
            ENDIF
            IF IN(36)=0 AND NTYPE AXIS(20)=0 AND NTYPE AXIS(21)=0 AND NTYPE AXIS(23)=0 _
                    AND VR(8)=8 AND (ABS(VR(1057)-VR(1061))<VR(177)) THEN'修正放置点
                IF VR(1003)=1 THEN
                    VR(445)=VR(421)+VR(407)-DPOS AXIS(40)-5
                    IF VR(445)>80 THEN
                        MOVELINK(INT((VR(1058)-VR(1054))*1000)/1000,VR(445),20,20,40,16)AXIS(36)',VR(432)
                        MOVELINK(INT((VR(1059)-VR(1055))*1000)/1000,VR(445),20,20,40,16)AXIS(37)
                        MOVELINK(VR(438),VR(408)+40,0.2*VR(408)+4,0.2*VR(408)+4,40,16+2,VR(422))AXIS(22)
'放置点下降
                        MOVELINK(INT((VR(1061)-VR(1057))*10000)/10000,VR(445),20,20,40,16)AXIS(38)
                        VR(8)=9
                        IF VR(0).19=1 AND IN(36)=0 THEN
                            VR(21).11=0
                        ENDIF
                    ELSE
                        VR(83).03=1
                    ENDIF
                ENDIF
            ENDIF
            'Z轴下降
        CASE 9
            IF DPOS AXIS(40)>VR(422) THEN
                MOVELINK(0-VR(438),VR(408)-1,0.2*VR(408),0.2*VR(408),40,16+2,VR(425))AXIS(22)
                PSWITCH(1,1,40,59,1,VR(418),VR(423))
                PSWITCH(0,1,40,60,1,VR(424),VR(425))
                IF VR(0).09=1 THEN
                    VR(0).04=1
                    VR(0).09=0
                ENDIF
                VR(8)=10
            ENDIF
        CASE 10
            VR(8)=11
        CASE 11
            IF DPOS AXIS(40)>=998 OR DPOS AXIS(40)<1 THEN
                VR(0).08=0
                VR(0).06=0
                VR(0).07=0
                VR(20).02=0
                VR(8)=0
                IF VR(7)=6 THEN
                    VR(7)=7
                ENDIF
                VR(4).16=0
            ENDIF
    END_CASE
    IF VR(3).15 OR (VR(0).01=1 AND VR(3).16=0) THEN
        OP(61,1)
    ENDIF
    IF VR(3).15 AND (VR(0).00=1 OR VR(29)=1) AND VR(3).16=0 THEN
        VR(30)=VR(30)+1
        VR(3).16=1
    ENDIF
    IF IN(37) THEN
        IF delay(12,500) THEN
            VR(3).15=1
        ENDIF
    ENDIF
    IF IN(37)=0 THEN
        OP(61,0)
        VR(3).15=0
    ENDIF
    IF VR(0).00=1 AND VR(0).01=0 AND VR(29)=1 THEN
        OP(62,1)
    ELSE
        OP(62,0)
    ENDIF

    IF rise(ABS(DPOS AXIS(3))<VR(173)/2,315) THEN
        OP(58,1)'滴胶
    ENDIF
    IF READ_OP(58)=1 THEN
        IF delay(13,VR(181)*1000) THEN
            OP(58,0)
        ENDIF
    ENDIF
    IF READ_OP(66)=1 THEN
        IF delay(14,VR(182)*1000) THEN
            OP(66,0)
        ENDIF
    ENDIF

'=====推板=====
    IF ABS(DPOS AXIS(3))<30 AND VR(111)=0 AND VR(21).09=1 AND (VR(9)=1 OR VR(9)=2) THEN
        VR(111)=1
    ENDIF
    SELECT_CASE VR(9)
        CASE 0
            IF VR(21).11=0 AND MTYPE AXIS(3)=0 AND (VR(3).15=1 OR VR(21).13=1) THEN
                IF ABS(MPOS AXIS(3))>1 THEN
                    IF VR(81).03=0 AND READ_OP(540)=0 THEN
                        MOVEABS(0) AXIS(3)'推板去
                        VR(9)=1
                    ENDIF
                ELSE
                    VR(9)=2
                ENDIF
            ENDIF
        CASE 1
            IF MTYPE AXIS(3)=0 AND VR(0).03=0 THEN
                IF ABS(DPOS AXIS(3))<1 THEN
                    VR(9)=2
                ELSE
                    IF VR(81).03=0 AND READ_OP(540)=0 THEN
                        MOVEABS(0) AXIS(3)'推板去
                    ENDIF
                ENDIF
            ENDIF
        CASE 2
            VR(119)=0-VR(173)
            IF VR(21).11=1 THEN
                VR(9)=3
            ENDIF
        CASE 3
            IF VR(21).11=1 AND MTYPE AXIS(3)=0 THEN
                IF (MPOS AXIS(3)-VR(119))>1 THEN
                    IF VR(81).03=0 AND READ_OP(540)=0 THEN
                        MOVEABS(VR(119)) AXIS(3)
                        VR(9)=4
                    ENDIF
                ELSE
                    VR(9)=5
                ENDIF
            ENDIF
        CASE 4
            IF MTYPE AXIS(3)=0 THEN
                IF ABS(MPOS AXIS(3)-VR(119))<1 THEN
                    VR(9)=5
                ELSE
                    IF VR(81).03=0 AND READ_OP(540)=0 THEN
                        MOVEABS(VR(119)) AXIS(3)
                    ENDIF
                ENDIF
            ENDIF
        CASE 5
            VR(9)=9
        CASE 9
            IF VR(111)=5 THEN
                VR(111)=0
            ENDIF
            IF VR(29)=1 OR VR(0).06 OR VR(0).07 OR VR(0).08 THEN
                IF MTYPE AXIS(3)=0 AND VR(82).14=0 AND VR(0).03=0 AND VR(2).10=0 AND VR(4).02=1 AND (VR(3).15=1 OR _
                        VR(21).13=1) THEN
                    VR(21).11=0
                    VR(4).02=0
                ENDIF
            ENDIF
            IF VR(21).11=0 THEN
                VR(9)=0
            ENDIF
    END_CASE

    IF VR(111)=1 OR VR(111)=2 THEN
        IF delay(15,VR(109)*1000)THEN
            VR(2).13=1
        ENDIF
    ENDIF
    SELECT_CASE VR(111)
        CASE 1
            VR(2).10=1
            VR(2).13=0
            IF delay(16,VR(107)*1000) THEN
                VR(111)=2
            ENDIF
        CASE 2
            VR(2).10=0
            IF VR(2).13=1 THEN
                VR(111)=3
            ENDIF
        CASE 3
            IF delay(17,VR(110)*1000) THEN
                VR(2).13=0
                VR(111)=4
            ENDIF
        CASE 4
            IF delay(18,200) THEN
                VR(111)=5
            ENDIF
    END_CASE

    IF VR(100)<>VR(91) THEN
        VR(92)=VR(24)
        VR(91)=VR(100)
    ENDIF
    IF (VR(24)-VR(92))>100 AND VR(100)>VR(88) THEN
        VR(88)=VR(100)
    ENDIF
    IF VR(4062)<> VR(93) THEN
        IF VR(86)<VR(85) THEN
            VR(86)=VR(85)
        ENDIF
        VR(85)=0
        VR(87)=0
        VR(93)=VR(4062)
    ENDIF
    IF VR(0).00=1 AND VR(0).01=0 THEN
        IF VR(4).18=0 THEN
            IF delay(19,35000) THEN
                VR(87)=VR(87)+0.01
                VR(4).18=1
            ENDIF
        ENDIF
    ENDIF
    IF VR(4).18=1 THEN
        IF delay(20,1000) THEN
            VR(4).18=0
        ENDIF
    ENDIF

WEND

