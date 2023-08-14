import pygame as pg
import numpy as np

####### DISPLAY CLASS : be careful with the init #######################################################
class cDisplay:
    background_color=(0,0,33)
    font_color=(33,150,33)
    font_size=23
    
    screenx=500
    screeny=500
    screencenterx=screenx/2
    screencentery=screeny/2
    screen = None
    font = None
    font2 = None

    fps=0
    fpscount=0
    stopwatchfps=0.0

    def __init__(self, init=True):
        if(init):
            pg.init()
            self.screen = pg.display.set_mode((self.screenx,self.screeny))
            pg.display.set_caption("TetrAI: AI plays Tetris")
        self.font = pg.font.SysFont("consolas", self.font_size, italic=False, bold=True)
        self.font2 = pg.font.SysFont("consolas", 15, italic=False, bold=False)
  
    # get delta time between two frames
    previousticks=pg.time.get_ticks()
    deltatime=0.0
    def __getdeltatime(self):
        ticks=pg.time.get_ticks()
        delta=0.001*(ticks-self.previousticks)
        self.previousticks=ticks
        self.deltatime=delta
        return delta
    
    # call this before drawing anything
    def update_begin(self):
        self.stopwatchfps+=self.deltatime
        self.fpscount+=1
        if self.stopwatchfps>=1.0:
            self.stopwatchfps=0.0
            self.fps=self.fpscount
            self.fpscount=0

        self.screen.fill(self.background_color)
        self.__quitevent()

    # call this after drawing all the stuff
    def update_end(self):
        self.__getdeltatime()
        pg.display.update()

    def get_keys_pressed():
        return pg.key.get_pressed()

    # exit the program, called in update_begin
    def __quitevent(self):
        for events in pg.event.get():
            if events.type==pg.QUIT:
                pg.quit()
                quit()

    # draw a textline on posx, posy
    def txt(self, textline, isbigFont=True,posx=0, posy=0):
        renderfont=self.font2.render(textline,True,self.font_color)
        if isbigFont:
            renderfont=self.font.render(textline,True,self.font_color)
        #textrect=renderfont.get_rect()
        textrect = (posx, posy)
        self.screen.blit(renderfont, textrect)

######## END OF DISPLAY CLASS #####################################################################

######## TETRIS CLASS #############################################################################

class cTetris:
    # line count and points
    linecount = 0
    points = 0
    level = 0

    linerecord =0
    pointsrecord=0

    # level changes every 10 lines
    levellines=0

    # actual block position
    actualBlockX = 0
    actualBlockY = 0

    # how long to wait until block moves down
    turntime_inittime = 0.750
    # gets adjusted when level changes
    turntime = turntime_inittime

    # how long to wait before polling for keys
    keytime = 0.1

    # the stop watches for the above variables.
    stopwatchturn=0.0
    stopwatchkeys=0.0

    # key simulated by the AI
    # 0 no key 1 left 2 right 3 up 4 down
    simulatedkey = 0

    # last key pressed by the player, as data for the AI
    lastplayerkey=0

    # constrain the up key to be pressed just once when key is down.
    upkeydown = False

    # is the game over?
    gameover = False

    # the blocks
    blockI = np.array([[0,0,0,0],
                       [1,1,1,1],
                       [0,0,0,0],
                       [0,0,0,0]])

    blockJ = np.array([[1,0,0],
                       [1,1,1],
                       [0,0,0]])

    blockL = np.array([[0,0,1],
                       [1,1,1],
                       [0,0,0]])

    blockO = np.array([[1,1],
                       [1,1]])

    blockS = np.array([[0,1,1],
                       [1,1,0],
                       [0,0,0]])

    blockT = np.array([[0,1,0],
                       [1,1,1],
                       [0,0,0]])

    blockZ = np.array([[1,1,0],
                       [0,1,1],
                       [0,0,0]])


    # the blocks assigned to a name.
    blocks = {"I": blockI,
              "J": blockJ,
              "L": blockL,
              "O": blockO,
              "S": blockS,
              "T": blockT,
              "Z": blockZ}

    # the actual block matrix which can be rotated.
    actualBlockMat=np.copy(blocks["O"])

    # the matrix of the next block
    nextBlockMat=np.copy(blocks["O"])
  
    # the playfield with all the blocks
    playfield = np.array([[0,0,0,0,0,0,0,0,0,0], #-2
                          [0,0,0,0,0,0,0,0,0,0], #-1
                          [0,0,0,0,0,0,0,0,0,0], #1
                          [0,0,0,0,0,0,0,0,0,0], #2
                          [0,0,0,0,0,0,0,0,0,0], #3
                          [0,0,0,0,0,0,0,0,0,0], #4
                          [0,0,0,0,0,0,0,0,0,0], #5
                          [0,0,0,0,0,0,0,0,0,0], #6
                          [0,0,0,0,0,0,0,0,0,0], #7
                          [0,0,0,0,0,0,0,0,0,0], #8
                          [0,0,0,0,0,0,0,0,0,0], #9
                          [0,0,0,0,0,0,0,0,0,0], #10
                          [0,0,0,0,0,0,0,0,0,0], #11
                          [0,0,0,0,0,0,0,0,0,0], #12
                          [0,0,0,0,0,0,0,0,0,0], #13
                          [0,0,0,0,0,0,0,0,0,0], #14 
                          [0,0,0,0,0,0,0,0,0,0], #15
                          [0,0,0,0,0,0,0,0,0,0], #16
                          [0,0,0,0,0,0,0,0,0,0], #17
                          [0,0,0,0,0,0,0,0,0,0], #18
                          [0,0,0,0,0,0,0,0,0,0], #19
                          [0,0,0,0,0,0,0,0,0,0]]) #20
    
    def __init__(self):
        # get a new block twice so the next block will be overwritten and the actual block is random.
        self.__getNextBlock()
        self.__getNextBlock()

    # get the block with this name
    def __getBlockMat(self,bname):
        return np.copy(self.blocks[bname])

    # get the next block
    def __getNextBlock(self):
        self.actualBlockY=0
        self.actualBlockX=4
        self.actualBlockMat = self.nextBlockMat

        rnd=np.random.randint(0,len(self.blocks))
        #print("Next Block: ",rnd)
        if rnd==0: 
            self.nextBlockMat = self.__getBlockMat("I")
        elif rnd==1: 
            self.nextBlockMat = self.__getBlockMat("J")
        elif rnd==2: 
            self.nextBlockMat = self.__getBlockMat("L")
        elif rnd==3: 
            self.nextBlockMat = self.__getBlockMat("O")
        elif rnd==4: 
            self.nextBlockMat = self.__getBlockMat("S")
        elif rnd==5: 
            self.nextBlockMat = self.__getBlockMat("T")
        elif rnd==6: 
            self.nextBlockMat = self.__getBlockMat("Z")
        else: 
            self.nextBlockMat = self.__getBlockMat("O")

    # clear all full lines
    def __clearLines(self):
        pt=0
        for line in range(len(self.playfield)):
            count = 0
            # count the values together
            for x in self.playfield[line]:
                count+=x

            # a full line counts 10
            if count>=10:
                #print("CLEAR LINE #", line)
                # add points and line count
                self.linecount+=1
                self.levellines+=1
                # level changed, adjust speed
                if self.levellines>=10:
                    self.levellines=0
                    self.level+=1
                    if self.turntime>0.1:
                        self.turntime -= 0.033

                pt+=1
                pt*=2
                pt+=self.level

                pf = np.copy(self.playfield)
                # move all the lines one down
                for l in range((len(pf))):
                    if l < line:
                        self.playfield[l+1]=pf[l]
                    # for lines below, leave original state
                    # (this else here does not work right)
                    #else:
                    #    self.playfield[l]=pf[l]
        
        # add the accumulated points
        self.points+=pt

        # maybe set record
        if self.linerecord<self.linecount:
            self.linerecord=self.linecount
        if self.pointsrecord<self.points:
            self.pointsrecord=self.points

    # check if new block position or rotation is valid on the playfield.
    def __checkForValidMove(self,mat,qx,qy):
        for maty in range(len(mat)):
            for matx in range(len(mat[maty])):
                #print(qx, qy, qx+mix, qy+miy)
                # check if field is set
                if mat[maty,matx]==1:
                    #print("ON")
                    # check if bounds are ok
                    if qx+matx>=0 and qy+maty>=0 and self.playfield.size/10>qy+maty:
                        # check if x bound is ok
                        if self.playfield[qy+maty].size > qx+matx:
                            # check if playfield at this position is set
                            if self.playfield[qy+maty, qx+matx]==1:
                                #print("PF SET")
                                return False
                        else:
                            #print("PF OUT OF X")
                            return False
                    else:
                        #print("PF OUT OF Y")
                        return False
        #print("--- OK ---")
        return True

    # rotate the actual block about 90degrees counterclockwise
    def rotate(self):
        mat=np.rot90(self.actualBlockMat)
        if self.__checkForValidMove(mat,self.actualBlockX,self.actualBlockY):
            self.actualBlockMat=mat

    # move the block
    laydown = False
    def move(self,x,y):
        if(self.__checkForValidMove(self.actualBlockMat,self.actualBlockX+x,self.actualBlockY+y)):
            self.actualBlockX+=x
            self.actualBlockY+=y
            self.laydown = False
        else:
            # dont lay down sidewards
            if(x != 0):
                self.laydown=False
            # lay down and incorporate the block into the playfield.
            if(y>=0):
                if(self.laydown):
                    #print("REAL STOP @", self.actualBlockX)
                    self.laydown = False
                    self.playfield=self.renderfield() # renderfield incorporates the actual block.
                    self.__getNextBlock()
                    self.__clearLines()
                else:
                    self.laydown=True
            #print("STOP @",self.actualBlockX, self.actualBlockY)

    # update the game, get key input, check stop watches, etc.
    def update(self):
        if self.gameover==True:
            return
        
        self.stopwatchturn+=dis.deltatime
        self.stopwatchkeys+=dis.deltatime


        # move the block one down after some time
        if(self.stopwatchturn>=tet.turntime):
            self.stopwatchturn=0.0
            tet.move(0,1)

        # constrain key presses to 1 each 100ms
        if(self.stopwatchkeys>self.keytime):
            self.lastplayerkey=0
            self.stopwatchkeys=0.0
            keys=pg.key.get_pressed()
            if keys[pg.K_LEFT] or keys[pg.K_a] or self.simulatedkey==1:
                self.move(-1,0)
                # maybe set lastplayerkey
                if(keys[pg.K_LEFT] or keys[pg.K_a]) and self.simulatedkey!=1:
                    self.lastplayerkey=1
            if keys[pg.K_RIGHT] or keys[pg.K_d] or self.simulatedkey==2:
                self.move(1,0)
                # maybe set lastplayerkey
                if(keys[pg.K_RIGHT] or keys[pg.K_d]) and self.simulatedkey!=2:
                    self.lastplayerkey=2
            if keys[pg.K_DOWN] or keys[pg.K_s] or self.simulatedkey==4:
                self.move(0,1)
                # maybe set lastplayerkey
                if(keys[pg.K_DOWN] or keys[pg.K_s]) and self.simulatedkey!=4:
                    self.lastplayerkey=4
            if ((keys[pg.K_UP] or keys[pg.K_w]) and not self.upkeydown) or self.simulatedkey==3:
                self.rotate()
                self.upkeydown = True
                # maybe set lastplayerkey
                if(keys[pg.K_UP] or keys[pg.K_w]) and self.simulatedkey!=3:
                    self.lastplayerkey=3
            # reset the up key blocker        
            if not keys[pg.K_UP] and not keys[pg.K_w]:
                self.upkeydown = False
            self.simulatedkey = 0

        # check for gameover and stop the game
        for x in self.playfield[1]:
            if x: # there is a block in the line on top
                self.gameover = True

    # render the playfield AND the actual block into a new matrix
    def renderfield(self):
        # copy the playfield
        renderedfield=np.copy(self.playfield)
        # draw the actual block matrix on the field
        for y in range(len(self.actualBlockMat)):
            for x in range(len(self.actualBlockMat[y])):
                if self.actualBlockY+y<renderedfield.size/10:
                    if self.actualBlockX+x<renderedfield[self.actualBlockY+y].size:
                        if self.actualBlockMat[y,x]:    
                            renderedfield[self.actualBlockY+y,self.actualBlockX+x]=1
                        else:
                            renderedfield[self.actualBlockY+y,self.actualBlockX+x]=self.playfield[self.actualBlockY+y, self.actualBlockX+x]
        return renderedfield

    # build the preview block text
    def buildpreview(self,mat):
        text=[]
        tex=""
        for mty in mat:
            tex=""
            for mtx in mty:
                if mtx:
                    tex+="[]"
                else:
                    tex+="  "
            text.append(tex)
        return text
    
    # build the playfield as text   
    def buildtextscreen(self, renderedfield):
        text=[]
        text.append("**********************")
        my=0
        mx=0
        # build the playfield in text
        for y in renderedfield:
            txt="*"
            mx=0
            if my>=2:
                for x in y:
                    if(x==1):
                        txt+="[]"
                    elif(x==2):
                        txt+="##" # Test Number / Field
                    else:
                        txt+=". "
                    #print(mx,my,self.playfield[my,mx], self.playfield.size, self.playfield[my].size)
                    mx+=1
                txt+="*"
                text.append(txt)
            my+=1  
        text.append("**********************")
        return text
    
    # reset the game
    def resetgame(self):
        # reset all values except for the records.
        self.points = 0
        self.linecount=0
        self.level = 0
        self.turntime=self.turntime_inittime
        for y in range(len(self.playfield)):
            for x in range(len(self.playfield[y])):
                self.playfield[y,x]=0
        self.__getNextBlock()
        self.gameover = False

############# END OF TETRIS CLASS #####################################################################################################

############# AI CLASS ################################################################################################################

class cTetrAI:
    scoreinitvalue = 20 # two lines for free
    recordscore = -100
    score = 0
    previousscore = 0
    deltascore = 0

    def reset(self):
        self.score = 0
        self.previousscore = 0
        self.deltascore = 0

    # process the AI stuff here
    def processAI(self, renderedfield, lines, points, gameover, lastplayerkey, nextblockmatrix, actualblockmatrix):
        # maybe the score for the AI weights is computed otherwise - idk
        sc = self.score
        # deltascore is score - record, so if it is 0, this is good. It will be < 0 elsewhere.
        self.score, self.deltascore = self.countscore(renderedfield,lines,points,gameover,lastplayerkey)

        # maybe set previous score
        if self.score!=sc:
            self.previousscore=sc
        # based on score, rf, lines, points, and gameover, and maybe lastplayerkey for learning; 
        #   compute the next key to press.

        # must be a value between 0 and 4
        # lastplayerkey uses the same values. 
        # So the AI can learn from a human beeing, maybe.
        # but it gets punished a little for using human help (in the countscore function) ;)
        
        ###########################
        #    ,-------------__     #
        #   {_   ___   ____--  *  #
        #    /  /,_|   |/         #
        #   /__/   |   |          #
        #           | |           #
        #           | |           #
        #           |_|           # 
        ####### COMPUTE HERE ######
        
        nextsimulatedkey = 0 # 1 left 2 right 3 rotate 4 speed up

        ########### ENDOF COMPUTE ##############

        # return the computed key so tetris can process the "input"
        return nextsimulatedkey
    
    def countscore(self,renderedfield,lines,points, gameover,lastplayerkey):
        # do the reward
        score = self.scoreinitvalue
        score+=lines*10
        score+=points*5

        # this values are doubled because of the record punishment below.
        # do the punishment
        for y in renderedfield:
            for x in y:
                if x: # it's set, bad for the ai
                    score-=1
        
        # player had to help, bad for the ai
        if lastplayerkey!=0:
            score-=10

        # it's game over, very bad for the ai
        if gameover:
            score-=100

        # keep record of the best score
        if score>=self.recordscore:
            # if it's better, award some points
            #score+=score-self.recordscore
            self.recordscore = score
        else:
            # if it's not better, punish the ai, maybe really hard.
            score -= (self.recordscore-score)
        return score, score-self.recordscore 

############# END OF AI CLASS #########################################################################################################

############# MAIN ####################################################################################################################

dis=cDisplay() # graphics and time adapter
tet=cTetris()  # the tetris game itself
ai = cTetrAI() # the AI playing tetris

while True:

    # update tetris
    tet.update()

    # rf is to use with the AI as input and for building up the screen.
    # it is a matrix with 0 and 1 values in size of the playfield.
    # also, the actual block is incorporated
    rf = tet.renderfield()

    # calculate the AI reaction for this turn.
    # where a turn is one frame and the must not move, but it can. 
    # Also, keypress is available all 100ms, not every frame.
    # The AI can "see" everything the player sees:
    # --> the playfield with the actual block in it (rf)
    # --> lines and points made
    # --> if it's gameover
    # --> the last key the player pressed or 0
    # --> preview (matrix) of the next block
    # --> matrix of the actual block. I think, this won't be needed because the block is already on rf.
    # this method should return the next key to press. 1 left 2 right 3 rotate 4 speed up

    # process AI turn or whatever and push the simulated key to tetris
    tet.simulatedkey=ai.processAI(rf, tet.linecount,tet.points, tet.gameover, tet.lastplayerkey,tet.nextBlockMat,tet.actualBlockMat)

    ####################### 
    # build up the screen #
    #######################

    # create a text with the playfield, border and actual block
    text = tet.buildtextscreen(rf)
    # create a text which shows the next block to come
    preview = tet.buildpreview(tet.nextBlockMat)

    # initialize the graphics turn
    dis.update_begin()

    # show the playfield on the screen
    for line in range(len(text)):
        dis.txt(text[line],True,0,dis.font_size*line)
    # show the preview on the screen
    for pline in range(len(preview)):
        dis.txt(preview[pline],True,320, 50+dis.font_size*pline)
    
    # show the tetris values on the screen
    dis.txt("Tetris Values:",False,295,125)

    text="Level: "+format(tet.level)
    dis.txt(text,False,320,140)

    text=format(tet.points)+" points"
    dis.txt(text,False,320,155)

    text=format(tet.linecount)+" lines"
    dis.txt(text,False,320,170)

    dis.txt("Tetris Records:",False,295, 195)

    text=format(tet.pointsrecord)+" points"
    dis.txt(text,False,320,210)

    text=format(tet.linerecord)+" lines"
    dis.txt(text, False, 320, 225)

    if tet.gameover:
        text="!! GAME OVER !!"
        dis.txt(text,True,295, 255)

    # show the AI values on the screen
    dis.txt("AI Values:",False, 295, 290)

    text= "Score:   "+format(ai.score)
    dis.txt(text,False, 320, 305)

    text = "> prev.: "+format(ai.previousscore)
    dis.txt(text, False,320,320)

    text = "> delta: "+format(ai.deltascore)
    dis.txt(text, False, 320, 335)

    text = "Record:  "+format(ai.recordscore)
    dis.txt(text, False,320,350)

    # show the key which the AI wishes to press
    text = "Press Key: "
    if tet.simulatedkey==0:
        text+="None"
    elif tet.simulatedkey==1:
        text+="<--"
    elif tet.simulatedkey==2:
        text+="-->"
    elif tet.simulatedkey==3:
        text+="Rotate"
    elif tet.simulatedkey==4:
        text+="Speed Up"
    dis.txt(text,False,320,365)

    text="FPS: "+format(dis.fps)
    dis.txt(text,False, 320, 390)

    #dis.txt("1 left 2 right 3 rotate 4 down",False,320,350)
    # end the drawing
    dis.update_end()

    # check for gameover and maybe restart
    if tet.gameover:
        pg.time.wait(5000)
        ai.reset()
        tet.resetgame()