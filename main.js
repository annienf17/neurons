/**
 * EGOEXPLAINED — NEURAL JOURNEY
 * Vanilla JavaScript + Three.js / WebGL. No React, no build tooling.
 *
 * Scroll controls an actual 3D camera; neural networks are procedurally generated.
 * Cursor's colourful fluid wake is a second lightweight 2D canvas.
 * Edit CONFIG to change scene density, speed/length, bloom and colors.
 */
import * as THREE from 'three';
import { EffectComposer } from 'three/addons/postprocessing/EffectComposer.js';
import { RenderPass } from 'three/addons/postprocessing/RenderPass.js';
import { UnrealBloomPass } from 'three/addons/postprocessing/UnrealBloomPass.js';
import { OutputPass } from 'three/addons/postprocessing/OutputPass.js';
import { mergeGeometries } from 'three/addons/utils/BufferGeometryUtils.js';

const CONFIG = {
  seed: 78941,
  journeyDistance: 440,   // how far the camera flies as the page scrolls (brain 0–250, body finale beyond, ends past the intestines)
  fogDensity: 0.0075,
  bloomStrength: 0.78,    // softer glow; colours stay saturated instead of burning to white
  bloomRadius: 0.55,
  bloomThreshold: 0.2,    // only bright cores/impulses bloom, not the whole frame
  neuronsDesktop: 320,
  neuronsMobile: 155,
  dustDesktop: 4200,
  dustMobile: 1850,
  maxPixelRatioDesktop: 1.45,
  maxPixelRatioMobile: 1.0,
};

const $ = (selector) => document.querySelector(selector);
const webglCanvas = $('#webgl');
const inkCanvas = $('#ink');
const stage = $('#stage');
const journey = $('#journey');
const loading = $('#loading');
const loadingText = $('#loading-text');
const progressFill = $('#progress-fill');
const progressLabel = $('#progress-label');
const depthLabel = $('#coordinates-depth');
const chapters = [...document.querySelectorAll('[data-chapter]')];
const jumpButtons = [...document.querySelectorAll('[data-jump]')];
const motionButton = $('#motion-toggle');
const motionLabel = $('#motion-label');
const cursorOrb = $('#cursor-orb');

const mobile = window.matchMedia('(max-width: 760px)').matches;
const prefersReducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)');
let reducedMotion = prefersReducedMotion.matches;
let targetProgress = 0;
let smoothProgress = 0;
let chapterIndex = -1;
let internalTime = 0;
let pointerX = 0;
let pointerY = 0;
let targetPointerX = 0;
let targetPointerY = 0;
let width = 1;
let height = 1;
let dpr = 1;

function seedRandom(seed) {
  let s = seed >>> 0;
  return () => {
    s = (s + 0x6d2b79f5) | 0;
    let t = Math.imul(s ^ (s >>> 15), 1 | s);
    t ^= t + Math.imul(t ^ (t >>> 7), 61 | t);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}
const random = seedRandom(CONFIG.seed);
const rand = (a, b) => a + (b - a) * random();
const clamp = (n, min, max) => Math.max(min, Math.min(max, n));
const mix = (a, b, t) => a + (b - a) * t;
const V = (x, y, z) => new THREE.Vector3(x, y, z);
const basePalette = [
  new THREE.Color('#fa297f'), new THREE.Color('#44eaff'),
  new THREE.Color('#9465ff'), new THREE.Color('#ffbf7d'),
  new THREE.Color('#5476ff'), new THREE.Color('#f054c8'),
];
const pickColor = () => basePalette[Math.floor(random() * basePalette.length)];

// =====================================
// THREE.JS / CINEMATIC WORLD
// =====================================
let renderer, scene, camera, composer;
let bloomPass;
let lineMaterial, particleMaterial;
let portalGroup;
// 0 → 1 as the camera passes through the brain: the network appears only inside it.
const networkFade = {value: 0};
// The glowing network is quieter in the body finale so the organs stay readable.
const networkDim = {value: 0};
// Chapter boundaries as flight depth (units travelled); chapter 05 also holds the body finale.
const CHAPTER_DEPTHS = [53.6, 107.2, 160.8, 214.4];
const CHAPTER_JUMPS = [0, 59, 115, 174, 230];
const fadeMaterials = [];   // plain materials whose opacity follows networkFade
let introTime = 0;          // seconds since start, drives the brain's materialise intro
let auroraMaterials = [];
let driftingGroup;
let clock = new THREE.Clock();
const lookSmooth = V(0, 0, -13);
const cameraSmooth = V(0, 0, 18);

function laneX(z) {
  return Math.sin(-z * 0.025) * 3.0 + Math.sin(-z * 0.063) * .9;
}
function laneY(z) {
  return Math.sin(-z * 0.038) * 1.3 + Math.cos(-z * 0.017) * .7;
}
function makeRenderer() {
  renderer = new THREE.WebGLRenderer({
    canvas: webglCanvas,
    antialias: false,
    alpha: false,
    depth: true,
    stencil: false,
    powerPreference: 'high-performance',
  });
  renderer.outputColorSpace = THREE.SRGBColorSpace;
  // Neutral tone mapping keeps bright magenta/cyan coloured instead of bleaching them to white.
  renderer.toneMapping = THREE.NeutralToneMapping;
  renderer.toneMappingExposure = 1.12;
  scene = new THREE.Scene();
  scene.background = new THREE.Color('#03050d');
  scene.fog = new THREE.FogExp2('#03050d', CONFIG.fogDensity);
  camera = new THREE.PerspectiveCamera(mobile ? 69 : 65, 1, 0.06, 200);
  camera.position.set(0, 0, 18);

  composer = new EffectComposer(renderer);
  composer.addPass(new RenderPass(scene, camera));
  bloomPass = new UnrealBloomPass(
    new THREE.Vector2(window.innerWidth, window.innerHeight),
    mobile ? .94 : CONFIG.bloomStrength,
    CONFIG.bloomRadius,
    CONFIG.bloomThreshold,
  );
  composer.addPass(bloomPass);
  composer.addPass(new OutputPass());
}

// Draw a curved neural fibre as many connected line segments. Packed into a
// SINGLE BufferGeometry for the entire scene, rather than thousands of draw calls.
function buildNeuralFibres() {
  const pos = [];
  const colours = [];
  const flows = [];
  const phases = [];
  const nodeLocations = [];
  const nodeColours = [];
  const addSegment = (p1, p2, c1, c2, a, b, phase) => {
    pos.push(p1.x, p1.y, p1.z, p2.x, p2.y, p2.z);
    colours.push(c1.r, c1.g, c1.b, c2.r, c2.g, c2.b);
    flows.push(a, b);
    phases.push(phase, phase);
  };
  function addBend(from, to, colour, phase, bends = 11, opacityFalloff = 0) {
    const delta = to.clone().sub(from);
    const drift = V(rand(-2.4, 2.4), rand(-2.1, 2.1), rand(-1.9, 1.9));
    const ctrl = from.clone().addScaledVector(delta, .5).add(drift);
    const curve = new THREE.QuadraticBezierCurve3(from, ctrl, to);
    const dark = colour.clone().multiplyScalar(mix(.55, .25, opacityFalloff));
    let previous = curve.getPoint(0);
    for (let j = 1; j <= bends; j++) {
      const t = j / bends;
      const next = curve.getPoint(t);
      const c0 = colour.clone().lerp(dark, t * .78);
      const c1 = colour.clone().lerp(dark, t * .85);
      addSegment(previous, next, c0, c1, (j - 1) / bends, t, phase);
      previous = next;
    }
  }
  const total = mobile ? CONFIG.neuronsMobile : CONFIG.neuronsDesktop;
  for (let i = 0; i < total; i++) {
    const z = rand(-CONFIG.journeyDistance - 38, 12);
    const theta = rand(0, Math.PI * 2);
    // Depth planes: ~30% foreground neurons hug the camera lane and fly past it,
    // the rest form the midground and the distant background.
    const tier = random();
    const radius = tier < .3 ? rand(3.2, 6.5) : tier < .75 ? rand(6.5, 14) : rand(14, 26);
    const centre = V(
      laneX(z) + Math.cos(theta) * radius,
      laneY(z) + Math.sin(theta) * radius * .75,
      z,
    );
    const colour = pickColor().clone();
    const secondary = basePalette[Math.floor(random() * basePalette.length)].clone();
    nodeLocations.push(centre);
    nodeColours.push(colour);
    const arms = Math.floor(rand(4, 8));
    for (let k = 0; k < arms; k++) {
      const azimuth = (k / arms) * Math.PI * 2 + rand(-.44, .44);
      const phi = rand(-1.1, 1.1);
      const length = rand(4, 11.5);
      const outward = V(
        Math.cos(azimuth) * Math.cos(phi),
        Math.sin(azimuth) * Math.cos(phi),
        Math.sin(phi),
      );
      const tip = centre.clone().addScaledVector(outward, length);
      tip.x += rand(-1.4, 1.4);
      tip.y += rand(-1, 1);
      const phase = rand(0, 50);
      addBend(centre, tip, colour, phase, 10);
      const forks = random() < .77 ? 2 : 1;
      for (let n = 0; n < forks; n++) {
        const twig = tip.clone().addScaledVector(outward, rand(2.4, 6.5));
        twig.add(V(rand(-3.3, 3.3), rand(-3.4, 3.4), rand(-2.8, 2.8)));
        addBend(tip, twig, colour.clone().lerp(secondary, .39), phase + .7 * n, 7, .5);
        if (random() < .34) {
          const tiny = twig.clone().add(V(rand(-2.9, 2.9), rand(-2.7, 2.7), rand(-2.7, 2.7)));
          addBend(twig, tiny, colour, phase + n, 5, .95);
        }
      }
    }
  }

  // Synaptic bridges: nearby somas connect into a recognizable, woven network.
  for (let i = 0; i < nodeLocations.length; i++) {
    const a = nodeLocations[i];
    let links = 0;
    for (let j = i + 1; j < nodeLocations.length && links < 3; j++) {
      const b = nodeLocations[j];
      const d = a.distanceTo(b);
      if (d > 7 && d < 23 && random() < .085) {
        addBend(a, b, nodeColours[i].clone().lerp(nodeColours[j], .5), rand(0, 50), 17, .42);
        links++;
      }
    }
  }

  // A few long helically winding neural highways emphasize forward depth.
  for (let lane = 0; lane < 9; lane++) {
    const angle = (lane / 9) * Math.PI * 2;
    const r = rand(7.6, 12.8);
    const colour = basePalette[lane % basePalette.length].clone().multiplyScalar(.64);
    for (let zz = 20; zz > -CONFIG.journeyDistance - 35; zz -= 1.65) {
      const coordinates = (z) => {
        const th = angle + z * (lane % 2 ? .026 : -.021);
        return V(laneX(z) + Math.cos(th) * (r + 1.4 * Math.sin(z * .035 + lane)),
          laneY(z) + Math.sin(th) * (r * .73), z);
      };
      addSegment(coordinates(zz), coordinates(zz - 1.65), colour, colour, zz / 40, (zz - 1.65) / 40, lane * 4.4);
    }
  }

  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute('position', new THREE.Float32BufferAttribute(pos, 3));
  geometry.setAttribute('color', new THREE.Float32BufferAttribute(colours, 3));
  geometry.setAttribute('aFlow', new THREE.Float32BufferAttribute(flows, 1));
  geometry.setAttribute('aPhase', new THREE.Float32BufferAttribute(phases, 1));
  lineMaterial = new THREE.ShaderMaterial({
    uniforms: {uTime: {value: 0}, uProgress: {value: 0}, uNetwork: networkDim},
    transparent: true, depthWrite: false, blending: THREE.AdditiveBlending,
    vertexColors: true,
    vertexShader: `
      attribute float aFlow;
      attribute float aPhase;
      varying vec3 vColor;
      varying float vFlow;
      varying float vPhase;
      varying float vDepth;
      uniform float uTime;
      void main() {
        vColor = color;
        vFlow = aFlow;
        vPhase = aPhase;
        vec3 p = position;
        p.xy += vec2(sin(uTime*.34 + p.z*.028 + aPhase), cos(uTime*.22 + p.z*.02 + aPhase)) * .055;
        vec4 mvPosition = modelViewMatrix * vec4(p, 1.0);
        vDepth = -mvPosition.z;
        gl_Position = projectionMatrix * mvPosition;
      }`,
    fragmentShader: `
      uniform float uTime;
      uniform float uProgress;
      uniform float uNetwork;
      varying vec3 vColor;
      varying float vFlow;
      varying float vPhase;
      varying float vDepth;
      void main() {
        float beat = pow(max(0.0, sin(vFlow * 13.0 - uTime * 2.65 + vPhase)), 16.0);
        float wave = .45 + .55 * sin(uTime * .65 + vPhase * 2.3);
        float visible = (1.0 - smoothstep(27.0, 137.0, vDepth)) * smoothstep(.6, 3.5, vDepth);
        // Cool cyan/blue grade from chapter 05, applied along the whole journey.
        vec3 shift = mix(vec3(1.0), vec3(.7,1.25,1.35), .46);
        // Impulses brighten in their own hue (slightly lavender) instead of flashing white.
        vec3 pulse = mix(vColor, vec3(.78,.7,1.0), .3) * beat * 1.45;
        float intensity = .5 + .28 * wave;
        gl_FragColor = vec4(vColor * shift * intensity + pulse, (0.3 + beat * .6) * visible * uNetwork);
      }`,
  });
  const network = new THREE.LineSegments(geometry, lineMaterial);
  network.frustumCulled = false;
  scene.add(network);
  return {nodeLocations, nodeColours};
}

function buildParticles(nodes) {
  const pos = [], colours = [], sizes = [], phases = [], kinds = [];
  function add(p, c, size, kind) {
    pos.push(p.x,p.y,p.z);
    colours.push(c.r,c.g,c.b);
    sizes.push(size);
    phases.push(rand(0,Math.PI*2));
    kinds.push(kind);
  }
  nodes.nodeLocations.forEach((p, i) => {
    add(p, nodes.nodeColours[i], rand(4.2, 7.5), 1);
    for(let j=0;j<3;j++) {
      add(p.clone().add(V(rand(-1.9,1.9),rand(-1.9,1.9),rand(-1.9,1.9))),
        nodes.nodeColours[i].clone().lerp(new THREE.Color('#c9b8ff'), .2), rand(1.8,3.6), .4);
    }
  });
  const count = mobile ? CONFIG.dustMobile : CONFIG.dustDesktop;
  for(let i=0;i<count;i++) {
    const z = rand(-CONFIG.journeyDistance-50, 22);
    const a = rand(0,Math.PI*2);
    const r = Math.sqrt(random()) * 41 + 1.5;
    const position = V(laneX(z) + Math.cos(a)*r,laneY(z)+Math.sin(a)*r*.8,z);
    const c = random() < .57 ? basePalette[1] : random()<.6?basePalette[2]:basePalette[0];
    add(position,c,rand(.9,3.4),0);
  }
  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute('position', new THREE.Float32BufferAttribute(pos, 3));
  geometry.setAttribute('color', new THREE.Float32BufferAttribute(colours, 3));
  geometry.setAttribute('aSize', new THREE.Float32BufferAttribute(sizes, 1));
  geometry.setAttribute('aPhase', new THREE.Float32BufferAttribute(phases, 1));
  geometry.setAttribute('aKind', new THREE.Float32BufferAttribute(kinds, 1));
  particleMaterial = new THREE.ShaderMaterial({
    uniforms: {uTime:{value:0},uPixelRatio:{value:1},uProgress:{value:0},uNetwork:networkDim},
    transparent:true,depthWrite:false,depthTest:true,vertexColors:true,
    blending:THREE.AdditiveBlending,
    vertexShader:`
      attribute float aSize;
      attribute float aPhase;
      attribute float aKind;
      uniform float uTime;
      uniform float uPixelRatio;
      varying vec3 vColor;
      varying float vPhase;
      varying float vDepth;
      varying float vKind;
      void main(){
        vColor = color;
        vPhase = aPhase;
        vKind = aKind;
        vec3 p = position;
        p.xy += vec2(sin(uTime*.34 + aPhase + p.z*.08), cos(uTime*.27 + aPhase + p.z*.05)) * (.06 + aKind * .05);
        vec4 mv = modelViewMatrix*vec4(p,1.0);
        vDepth = -mv.z;
        float twinkle = .76 + .24 * sin(uTime*2.1 + aPhase*3.8);
        gl_PointSize = clamp(uPixelRatio * aSize * 30.0 / max(1.0,-mv.z) * twinkle, 1.0, 34.0*uPixelRatio);
        gl_Position=projectionMatrix*mv;
      }`,
    fragmentShader:`
      uniform float uTime;
      uniform float uProgress;
      uniform float uNetwork;
      varying vec3 vColor;
      varying float vPhase;
      varying float vDepth;
      varying float vKind;
      void main(){
        vec2 p = gl_PointCoord * 2.0 - 1.0;
        float r = length(p);
        if(r>1.0) discard;
        float halo = pow(max(0.0,1.0-r), 2.9);
        float core = pow(max(0.0,1.0-r), 16.0);
        float fog = (1.0 - smoothstep(28.0, 140.0, vDepth)) * smoothstep(.8, 4.0, vDepth);
        float shine = .77 + .23*sin(uTime*1.4+vPhase*5.0);
        float a = (halo * .55 + core*.45) * fog * shine;
        // Bright centre keeps its own colour (no white-out).
        gl_FragColor = vec4(vColor*(.72+core*.55),a*uNetwork);
      }`,
  });
  const points = new THREE.Points(geometry,particleMaterial);
  points.frustumCulled = false;
  scene.add(points);

  // A few actual 3D structures; highlights reveal their faceted contours as you fly past.
  const ico = new THREE.IcosahedronGeometry(.37, 1);
  const somaMaterial = new THREE.MeshBasicMaterial({color:0xffffff,vertexColors:false,transparent:true,opacity:0,blending:THREE.AdditiveBlending,depthWrite:false});
  somaMaterial.userData.baseOpacity = .28;
  fadeMaterials.push(somaMaterial);
  const instanced = new THREE.InstancedMesh(
    ico,
    somaMaterial,
    Math.min(nodes.nodeLocations.length, mobile?65:115),
  );
  const dummy = new THREE.Object3D();
  for(let i=0;i<instanced.count;i++) {
    dummy.position.copy(nodes.nodeLocations[i]);
    dummy.scale.setScalar(rand(.3,1.1));
    dummy.rotation.set(rand(0,3),rand(0,3),rand(0,3));
    dummy.updateMatrix();
    instanced.setMatrixAt(i,dummy.matrix);
    instanced.setColorAt(i,nodes.nodeColours[i]);
  }
  instanced.instanceMatrix.needsUpdate=true;
  if(instanced.instanceColor) instanced.instanceColor.needsUpdate=true;
  scene.add(instanced);
}

// Procedural colour veils, no textures/videos/assets required.
function buildAuroras() {
  for(let i=0;i<7;i++) {
    const z = 2-i*49;
    const geometry=new THREE.PlaneGeometry(120,80,1,1);
    const material = new THREE.ShaderMaterial({
      uniforms:{uTime:{value:0},uProgress:{value:0},uSeed:{value:i*1.7}},
      vertexShader:`
        varying vec2 vUv;
        void main(){vUv=uv;gl_Position=projectionMatrix*modelViewMatrix*vec4(position,1.0);}`,
      fragmentShader:`
        uniform float uTime;
        uniform float uProgress;
        uniform float uSeed;
        varying vec2 vUv;
        void main(){
          vec2 uv=vUv;
          float bend=sin(uv.x*7.0+uSeed+uTime*.13)*.10
                    +sin(uv.x*13.0-uSeed*.7-uTime*.08)*.034;
          float y=uv.y-.5-bend;
          float b1=exp(-pow(y-.11,2.0)*31.0);
          float b2=exp(-pow(y+.18+sin(uv.x*4.0+uTime*.12)*.08,2.0)*35.0);
          float wisps=.65+.35*sin(uv.x*20.0 + sin(uv.y*11.0)*3.0 + uSeed + uTime*.19);
          float edge=pow(sin(uv.x*3.14159),1.7)*pow(sin(uv.y*3.14159),1.5);
          // Blue / indigo veils only — no pink clouds at any stage of the journey.
          vec3 blue=vec3(.025,.36,.90), indigo=vec3(.16,.10,.80);
          vec3 c=mix(indigo,blue,clamp(uv.x+sin(uSeed)*.25,0.0,1.0));
          float a=(b1*.10+b2*.08)*edge*wisps;
          gl_FragColor=vec4(c*1.2,a);
        }`,
      transparent:true,blending:THREE.AdditiveBlending,depthWrite:false,depthTest:true,
      side:THREE.DoubleSide,
    });
    const plane=new THREE.Mesh(geometry,material);
    plane.position.set(rand(-9,9),rand(-2,2),z);
    plane.rotation.z=rand(-.25,.25);
    scene.add(plane);
    auroraMaterials.push(material);
  }
}

function buildPortals() {
  portalGroup=new THREE.Group();
  for(let i=0;i<9;i++) {
    const z= -26-i*30.5;
    const radius=rand(4.8,8.5);
    const colour=basePalette[i%basePalette.length].clone();
    for(let ring=0;ring<2;ring++){
      const n=160;
      const vertices=[];
      const ringRadius=radius+(ring*.75);
      for(let j=0;j<=n;j++){
        const t=(j/n)*Math.PI*2;
        const wave=Math.sin(t*6+i*2)*.3 + Math.cos(t*9+i)*.1;
        vertices.push(
          laneX(z)+Math.cos(t)* (ringRadius+wave),
          laneY(z)+Math.sin(t)* (ringRadius+wave)*.8,
          z+Math.sin(3*t+i)*2.1+ring*.8,
        );
      }
      const lineGeometry=new THREE.BufferGeometry();
      lineGeometry.setAttribute('position',new THREE.Float32BufferAttribute(vertices,3));
      const ringMaterial=new THREE.LineBasicMaterial({
        color:colour,transparent:true,opacity:0,
        blending:THREE.AdditiveBlending,depthWrite:false,
      });
      ringMaterial.userData.baseOpacity=ring===0?.31:.13;
      fadeMaterials.push(ringMaterial);
      portalGroup.add(new THREE.Line(lineGeometry,ringMaterial));
    }
  }
  scene.add(portalGroup);
}

// =====================================
// HOLOGRAPHIC BRAIN — revealed only in chapter 05
// =====================================
// Glassy, transparent brain built from real 3D geometry: two hemispheres with
// a longitudinal fissure, temporal lobes, cerebellum and brain stem. Glowing
// sulci lines follow the actual grooves; the neural network shines through it.
let brainGroup, brainMaterial;
const BRAIN_OPACITY = .1;   // how much the brain's shadow hides the network behind it (0 = none, 1 = solid)
const BRAIN_TISSUE = .45;   // brightness of the ivory tissue colour (0 = invisible, 1 = full)
const smooth01 = (a, b, x) => { const t = clamp((x - a) / (b - a), 0, 1); return t * t * (3 - 2 * t); };
function brainFold(x, y, z) {
  // Domain-warped field: its zero contours form dense, meandering, worm-like gyri.
  const wx = x + .12 * Math.sin(y * 9 + z * 5) + .06 * Math.sin(z * 17 + x * 3);
  const wy = y + .12 * Math.sin(z * 8 + x * 7) + .06 * Math.sin(x * 15);
  const wz = z + .12 * Math.sin(x * 8 + y * 6);
  return Math.sin(wx * 15 + Math.sin(wy * 11) * 1.8) * Math.cos(wy * 14 + Math.sin(wz * 12) * 1.8)
    + .6 * Math.sin(wz * 16 + wx * 6 + Math.sin(wy * 9) * 2);
}
function brainPart(widthSegments, heightSegments, part, shape) {
  const geometry = new THREE.SphereGeometry(1, widthSegments, heightSegments);
  const position = geometry.attributes.position;
  const folds = new Float32Array(position.count);
  const parts = new Float32Array(position.count).fill(part);
  const d = V(0, 0, 0), p = V(0, 0, 0);
  for (let i = 0; i < position.count; i++) {
    d.fromBufferAttribute(position, i);
    folds[i] = shape(d, p);
    position.setXYZ(i, p.x, p.y, p.z);
  }
  geometry.setAttribute('aFold', new THREE.BufferAttribute(folds, 1));
  geometry.setAttribute('aPart', new THREE.BufferAttribute(parts, 1));
  geometry.computeVertexNormals();
  return geometry;
}
function buildBrain() {
  const hi = !mobile;
  // Cerebrum: x = front/back, y = up, z = left/right.
  // Dense mesh so the narrow sulci between the many gyri are actually resolved.
  const cerebrum = brainPart(hi ? 340 : 170, hi ? 240 : 120, 0, (d, p) => {
    let x = d.x, y = d.y * .74, z = d.z * .86;
    const lower = Math.max(0, -d.y);
    const lateral = smooth01(.2, .75, Math.abs(d.z));
    // Occipital pole: lower and gently tapered at the back.
    if (x < -.5) { const t = (-.5 - x) / .5; y *= 1 - t * .2; z *= 1 - t * .14; }
    if (x < 0) y -= -x * .05;
    if (y < 0) y *= .55;                                        // flattened base
    // Frontal lobe: its underside (orbital surface) sits higher than the temporal pole.
    y += smooth01(.3, .8, x) * lower * .14;
    // Temporal lobe hangs below the middle of the hemisphere, its pole pointing forward.
    const temporal = Math.exp(-((x - .08) ** 2) / .11) * lower * lateral;
    y -= temporal * .24;
    // Lateral (Sylvian) fissure: deep groove separating the temporal lobe.
    const sylvian = y - (-.06 + (.3 - x) * .32);
    const sylvianCut = Math.exp(-sylvian * sylvian / .0028) * smooth01(-.45, -.25, x) * (1 - smooth01(.45, .62, x)) * lateral;
    // Longitudinal fissure between the hemispheres.
    const fissure = Math.exp(-(d.z * d.z) / .004) * smooth01(-.15, .45, d.y);
    y -= fissure * .14;
    const fold = brainFold(x, y, z);
    // Anatomical sulci with a fixed course: central sulcus with the parallel
    // pre-/postcentral sulci, and the superior temporal sulcus below the Sylvian fissure.
    const above = smooth01(-.02, .04, sylvian);
    const dC = x - (.22 - y * .38) - Math.sin(y * 17 + 1.3) * .022 - Math.sin(y * 6) * .018;
    const central = Math.exp(-dC * dC / .0012) * above;
    const precentral = (Math.exp(-((dC - .13) ** 2) / .0009) + Math.exp(-((dC + .12) ** 2) / .0009)) * above * .7;
    const sts = Math.exp(-((sylvian + .12) ** 2) / .0009) * (1 - smooth01(-.02, .02, sylvian))
      * smooth01(-.5, -.3, x) * (1 - smooth01(.4, .55, x)) * lateral;
    // Gyri are rounded bulges; sulci are narrow and deep between them.
    const gyrus = 1 - Math.exp(-fold * fold / .06);
    const k = .968 + gyrus * .046 - sylvianCut * .08 - (central + precentral + sts) * .045;
    p.set(x * k, y * k, z * k);
    return fold;
  });
  // Cerebellum: tucked under the occipital lobe, behind the brain stem.
  const cerebellum = brainPart(hi ? 120 : 70, hi ? 110 : 64, 1, (d, p) => {
    const fold = Math.sin(d.y * 34 + Math.sin(d.z * 3) * 1.2) * 1.4;
    const k = 1 - Math.exp(-fold * fold / .05) * .06;
    const flat = d.y > 0 ? .7 : 1;      // flattened top under the cerebrum
    p.set(-.54 + d.x * .3 * k, -.4 + d.y * .19 * k * flat, d.z * .5 * k);
    return fold;
  });
  // Brain stem: emerges from under the centre and descends slightly backwards.
  const stem = brainPart(32, 24, 2, (d, p) => {
    const taper = 1 - Math.max(0, -d.y) * .25;
    p.set(-.22 + d.x * .1 * taper + d.y * .07, -.6 + d.y * .3, d.z * .1 * taper);
    return 3;   // no sulci lines on the stem
  });

  brainMaterial = new THREE.ShaderMaterial({
    uniforms: {
      uTime: {value: 0},
      uReveal: {value: 0},
      uFade: {value: 1},     // fades the whole brain out once the camera is inside
      // Ivory brain tissue: warm ivory crests, soft cream and light taupe in the depths.
      uPale: {value: new THREE.Color('#fff8e7')},
      uRose: {value: new THREE.Color('#e9dcc2')},
      uMauve: {value: new THREE.Color('#d6c8b0')},
    },
    // Depth-tested front faces only: the brain reads as one solid glass shell
    // instead of overlapping bubbles (hidden cerebellum/back walls no longer shine through).
    transparent: true, depthWrite: false, depthTest: true,
    blending: THREE.AdditiveBlending, side: THREE.FrontSide,
    vertexShader: `
      attribute float aFold;
      attribute float aPart;
      varying vec3 vNormal;
      varying vec3 vView;
      varying vec3 vLocal;
      varying float vFold;
      varying float vPart;
      varying float vNear;
      void main() {
        vLocal = position;
        vFold = aFold;
        vPart = aPart;
        vec4 mv = modelViewMatrix * vec4(position, 1.0);
        vNormal = normalize(normalMatrix * normal);
        vView = normalize(-mv.xyz);
        // Surface dissolves softly just before the lens as the camera flies through it.
        vNear = smoothstep(1.2, 6.0, -mv.z);
        gl_Position = projectionMatrix * mv;
      }`,
    fragmentShader: `
      uniform float uTime;
      uniform float uReveal;
      uniform float uFade;
      uniform vec3 uPale;
      uniform vec3 uRose;
      uniform vec3 uMauve;
      varying float vNear;
      varying vec3 vNormal;
      varying vec3 vView;
      varying vec3 vLocal;
      varying float vFold;
      varying float vPart;
      // Anti-aliased line where a boundary function crosses zero.
      float boundary(float d) {
        return 1.0 - smoothstep(0.0, fwidth(d) * 2.2 + .002, abs(d));
      }
      void main() {
        // Materialise from the base upwards, with a soft glowing edge.
        float edge = uReveal * 2.0 - .95;
        float shown = 1.0 - smoothstep(edge - .08, edge + .08, vLocal.y);
        if (shown <= .001) discard;
        float edgeGlow = exp(-pow((vLocal.y - edge) * 14.0, 2.0)) * (1.0 - smoothstep(.85, 1.0, uReveal));
        float facing = gl_FrontFacing ? 1.0 : .45;
        float fres = pow(1.0 - abs(dot(normalize(vNormal), vView)), 2.4);
        // Sulci: thin, anti-aliased contour lines exactly along the carved grooves.
        float w = fwidth(vFold) * 1.3;
        float sulci = 1.0 - smoothstep(0.0, w, abs(vFold));
        // Slow light scan sweeping front to back, plus impulses running along the folds.
        float scan = exp(-pow((vLocal.x - (mod(uTime * .22, 2.8) - 1.4)) * 5.0, 2.0));
        float spark = pow(max(0.0, sin(vFold * 3.0 + vLocal.x * 6.0 - uTime * 1.6)), 12.0);
        vec3 colour = mix(uRose, uPale, smoothstep(-.6, .8, vLocal.x));
        colour = mix(colour, uMauve, smoothstep(.1, .8, vLocal.y) * .55);

        // Anatomical regions (x = front/back, y = up). Major landmarks follow
        // their real, gently meandering course instead of straight seams.
        float x = vLocal.x, y = vLocal.y;
        float sylvianY = -.06 + (.3 - x) * .32 + sin(x * 13.0) * .012;   // lateral (Sylvian) fissure
        float dSylvian = y - sylvianY;
        float dCentral = x - (.22 - y * .38) - sin(y * 17.0 + 1.3) * .022 - sin(y * 6.0) * .018; // central sulcus
        float dOccipital = x - y * .12 + .55 + sin(y * 12.0) * .015;     // parieto-occipital boundary
        float cerebrum = vPart < .5 ? 1.0 : 0.0;
        float below = 1.0 - smoothstep(-.03, .03, dSylvian);
        float occipital = 1.0 - smoothstep(-.04, .04, dOccipital);
        float frontal = smoothstep(-.03, .03, dCentral) * (1.0 - below) * (1.0 - occipital);
        float temporal = below * (1.0 - occipital) * smoothstep(-.6, -.45, x);
        float parietal = max(0.0, 1.0 - frontal - temporal - occipital);
        vec3 lobe = uPale * frontal + uMauve * parietal + uRose * temporal + mix(uMauve, uRose, .5) * occipital;
        lobe = mix(mix(uPale, uRose, .35), lobe, cerebrum);   // cerebellum
        lobe = mix(lobe, uMauve * .8, vPart > 1.5 ? 1.0 : 0.0); // brain stem
        float lines = cerebrum * (
            boundary(dSylvian) * (1.0 - smoothstep(.4, .5, x)) * smoothstep(-.45, -.35, x)
          + boundary(dCentral) * smoothstep(sylvianY, sylvianY + .05, y)
          + boundary(dOccipital) * smoothstep(sylvianY - .25, sylvianY - .1, y));
        float outline = smoothstep(.55, .9, fres);       // crisp silhouette of the whole brain
        // Soft key light from above-front sculpts the gyri as real relief.
        vec3 n = normalize(vNormal);
        vec3 L = normalize(vec3(.35, .75, .55));
        float diffuse = max(dot(n, L), 0.0) * .65 + .35;   // wrapped light, like soft tissue
        float sheen = pow(max(dot(n, normalize(L + vView)), 0.0), 24.0);
        // Gyri crests catch the light; sulci and fissures sink into darker, rosier tissue.
        float crest = smoothstep(.15, .9, abs(vFold));
        vec3 tissue = mix(mix(colour, uRose, .35), mix(colour, uPale, .45), crest);
        tissue = mix(tissue, lobe, .25);

        vec3 col = tissue * (.045 + diffuse * .13 + fres * .2)
                 + uPale * sheen * .05
                 + uPale * scan * crest * .025
                 + uPale * outline * .08
                 + uPale * edgeGlow * .3;
        col *= (1.0 - sulci * .45) * (1.0 - min(lines, 1.0) * .4);
        // Lighter tissue in the interior, crisper at the silhouette: a see-through brain.
        float body = mix(.55, 1.0, fres);
        gl_FragColor = vec4(col * body * ${BRAIN_TISSUE.toFixed(2)} * shown * facing * vNear * uFade * smoothstep(0.0, .12, uReveal), 1.0);
      }`,
  });
  brainGroup = new THREE.Group();
  // Shadow pass: a soft dark silhouette that dims the glowing network behind it.
  const shadowMaterial = new THREE.ShaderMaterial({
    uniforms: {uReveal: brainMaterial.uniforms.uReveal, uFade: brainMaterial.uniforms.uFade},
    transparent: true, depthWrite: false, depthTest: true,
    blending: THREE.NormalBlending, side: THREE.FrontSide,
    vertexShader: brainMaterial.vertexShader,
    fragmentShader: `
      uniform float uReveal;
      uniform float uFade;
      varying float vNear;
      varying vec3 vNormal;
      varying vec3 vView;
      varying vec3 vLocal;
      varying float vFold;
      void main() {
        float edge = uReveal * 2.0 - .95;
        float shown = 1.0 - smoothstep(edge - .08, edge + .08, vLocal.y);
        float fres = pow(1.0 - abs(dot(normalize(vNormal), vView)), 1.6);
        // Densest in the middle, feathered towards the outline like a real shadow.
        float alpha = (1.0 - fres * .6) * ${BRAIN_OPACITY.toFixed(2)} * shown * vNear * uFade * smoothstep(0.0, .2, uReveal);
        gl_FragColor = vec4(vec3(.02, .018, .012), alpha);
      }`,
  });
  // Depth pre-pass: writes only depth, so each pixel shows just the nearest brain surface.
  const depthMaterial = new THREE.MeshBasicMaterial({colorWrite: false, depthWrite: true, transparent: true});
  for (const geometry of [cerebrum, cerebellum, stem]) {
    const depth = new THREE.Mesh(geometry, depthMaterial);
    depth.renderOrder = 1;
    const shadow = new THREE.Mesh(geometry, shadowMaterial);
    shadow.renderOrder = 2;
    const outline = new THREE.Mesh(geometry, brainMaterial);
    outline.renderOrder = 3;
    brainGroup.add(depth, shadow, outline);
  }
  // Chapter 01: the brain greets the visitor right in front of the camera; the
  // camera's path runs straight through it, so the first scroll flies inside.
  const z = 0;
  brainGroup.position.set(laneX(z) + (mobile ? 0 : 3), laneY(z) + (mobile ? 2 : .5), z);
  brainGroup.scale.setScalar(mobile ? 7 : 9);
  brainGroup.rotation.set(.1, -.22, 0);
  brainGroup.visible = false;
  brainGroup.renderOrder = 2;
  scene.add(brainGroup);
}

// =====================================
// INSIDE THE BRAIN — cortex walls + anatomical structures per chapter
// =====================================
// After the camera passes through the brain, the journey continues inside it:
// folded cortex walls enclose the network, and each chapter flies past a real
// structure in the same translucent ivory style, with a discreet 3D label.
const INTERIOR_TISSUE = .5;   // brightness of the interior structures (0 = invisible, 1 = full)
const interiorMaterials = [];
const brainLabels = [];

function makeTissueMaterial(side, intensity, fogFar) {
  const material = new THREE.ShaderMaterial({
    uniforms: {
      uTime: brainMaterial.uniforms.uTime,
      uNetwork: networkFade,
      uPale: brainMaterial.uniforms.uPale,
      uRose: brainMaterial.uniforms.uRose,
      uIntensity: {value: intensity * INTERIOR_TISSUE},
      uFogFar: {value: fogFar},
    },
    transparent: true, depthWrite: false, depthTest: false,
    blending: THREE.AdditiveBlending, side,
    vertexShader: `
      attribute float aFold;
      varying vec3 vNormal;
      varying vec3 vView;
      varying float vFold;
      varying float vDepth;
      void main() {
        vFold = aFold;
        vec4 mv = modelViewMatrix * vec4(position, 1.0);
        vNormal = normalize(normalMatrix * normal);
        vView = normalize(-mv.xyz);
        vDepth = -mv.z;
        gl_Position = projectionMatrix * mv;
      }`,
    fragmentShader: `
      uniform float uTime;
      uniform float uNetwork;
      uniform vec3 uPale;
      uniform vec3 uRose;
      uniform float uIntensity;
      uniform float uFogFar;
      varying vec3 vNormal;
      varying vec3 vView;
      varying float vFold;
      varying float vDepth;
      void main() {
        float fog = 1.0 - smoothstep(uFogFar * .3, uFogFar, vDepth);
        float near = smoothstep(.8, 3.5, vDepth);
        if (fog * near <= .001) discard;
        vec3 n = normalize(vNormal);
        if (!gl_FrontFacing) n = -n;
        float fres = pow(1.0 - abs(dot(n, vView)), 2.2);
        float diffuse = max(dot(n, normalize(vec3(.35, .75, .55))), 0.0) * .6 + .4;
        // Fold lines sink darker; crests catch the light — like real tissue.
        float line = 1.0 - smoothstep(0.0, fwidth(vFold) * 1.4, abs(vFold));
        float crest = smoothstep(.2, .9, abs(vFold));
        vec3 tissue = mix(uRose, uPale, crest * .6 + .2);
        // A faint signal wave travelling through the tissue towards the camera.
        float signal = pow(max(0.0, sin(vDepth * .25 + uTime * 1.4)), 10.0);
        vec3 col = tissue * (.02 + diffuse * .1 + fres * .22) * (1.0 - line * .4) + uPale * line * signal * .06;
        gl_FragColor = vec4(col * fog * near * uNetwork * uIntensity, 1.0);
      }`,
  });
  interiorMaterials.push(material);
  return material;
}

// Tube along a curve with a radius/fold profile: profile(u, angle) → [radius, fold].
// u runs 0 → 1 along the curve, angle 0 → 2π around it.
function profiledTube(points, profile, radial = 32) {
  const curve = new THREE.CatmullRomCurve3(points);
  const radialSegments = mobile ? 20 : radial;
  const geometry = new THREE.TubeGeometry(curve, mobile ? 130 : 240, 1, radialSegments, false);
  const position = geometry.attributes.position;
  const uv = geometry.attributes.uv;
  const folds = new Float32Array(position.count);
  const centre = V(0, 0, 0), offset = V(0, 0, 0);
  for (let i = 0; i < position.count; i++) {
    const u = uv.getX(i);
    const angle = ((i % (radialSegments + 1)) / radialSegments) * Math.PI * 2;
    curve.getPointAt(u, centre);
    const [r, fold] = profile(u, angle);
    offset.fromBufferAttribute(position, i).sub(centre).multiplyScalar(r).add(centre);
    position.setXYZ(i, offset.x, offset.y, offset.z);
    folds[i] = fold;
  }
  geometry.setAttribute('aFold', new THREE.BufferAttribute(folds, 1));
  geometry.computeVertexNormals();
  return geometry;
}
// Rounded closing of a tube at both ends.
const tubeCaps = (u, front = 10, back = 12) =>
  Math.sqrt(Math.sin(Math.min(1, u * front) * Math.PI * .5) * Math.sin(Math.min(1, (1 - u) * back) * Math.PI * .5));
const bump = (u, centre, width) => Math.exp(-(((u - centre) / width) ** 2));

function lanePoint(z, dx, dy) {
  return V(laneX(z) + dx, laneY(z) + dy, z);
}

// Elegant single-line 3D label drawn into a canvas texture.
function addBrainLabel(title, position, far = 55) {
  const canvas = document.createElement('canvas');
  canvas.width = 1024;
  canvas.height = 160;
  const texture = new THREE.CanvasTexture(canvas);
  texture.colorSpace = THREE.SRGBColorSpace;
  const draw = () => {
    const ctx = canvas.getContext('2d');
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.fillStyle = 'rgba(255,248,231,.92)';
    ctx.beginPath();
    ctx.arc(14, 80, 6, 0, Math.PI * 2);
    ctx.fill();
    ctx.fillRect(30, 79, 60, 2);
    let x = 108;
    ctx.font = "700 40px Manrope, 'DM Sans', Arial, sans-serif";
    for (const ch of title) { ctx.fillText(ch, x, 94); x += ctx.measureText(ch).width + 7; }
    texture.needsUpdate = true;
  };
  draw();
  document.fonts?.ready.then(draw);
  const sprite = new THREE.Sprite(new THREE.SpriteMaterial({
    map: texture, transparent: true, opacity: 0, depthWrite: false, depthTest: false, fog: false, toneMapped: false,
  }));
  sprite.position.copy(position);
  sprite.scale.set(mobile ? 9 : 12, (mobile ? 9 : 12) * 160 / 1024, 1);
  sprite.center.set(0, .5);
  sprite.renderOrder = 5;
  sprite.userData.far = far;   // distance at which the label has fully faded out
  scene.add(sprite);
  brainLabels.push(sprite);
}

function buildInterior() {
  const group = new THREE.Group();

  // White matter: inside the brain we travel through bundles of nerve fibres,
  // so the enclosing walls carry long, gently weaving fibre striations.
  const rings = mobile ? 150 : 280, seg = mobile ? 56 : 104;
  const pos = [], folds = [], index = [];
  for (let i = 0; i < rings; i++) {
    const z = mix(-4, -262, i / (rings - 1));
    // Opens wide right behind the brain's surface, and again where we leave through the brainstem.
    const mouth = 1 + smooth01(-26, -4, z) * .8 + smooth01(236, 262, -z) * .9;
    for (let j = 0; j <= seg; j++) {
      const t = (j / seg) * Math.PI * 2;
      const r = (24 + 3 * Math.sin(t * 2 + z * .04) + 2 * Math.sin(t * 5 - z * .07)) * mouth;
      const fibre = Math.sin(t * 38 + Math.sin(z * .05 + t * 3) * 1.4);
      const rr = r - fibre * .35;                   // fibre bundles give a soft ribbed relief
      pos.push(laneX(z) + Math.cos(t) * rr, laneY(z) + Math.sin(t) * rr * .8, z);
      folds.push(fibre);
      if (i < rings - 1 && j < seg) {
        const a = i * (seg + 1) + j, b = a + seg + 1;
        index.push(a, b, a + 1, b, b + 1, a + 1);
      }
    }
  }
  const whiteMatter = new THREE.BufferGeometry();
  whiteMatter.setAttribute('position', new THREE.Float32BufferAttribute(pos, 3));
  whiteMatter.setAttribute('aFold', new THREE.Float32BufferAttribute(folds, 1));
  whiteMatter.setIndex(index);
  whiteMatter.computeVertexNormals();
  group.add(new THREE.Mesh(whiteMatter, makeTissueMaterial(THREE.DoubleSide, .55, 72)));
  addBrainLabel('WHITE MATTER', lanePoint(-22, -11, 7));

  // Chapter 02 — corpus callosum. Seen from the side it is a "C": thin rostrum
  // curling under the genu at the front, a long arched body, and the thick,
  // bulbous splenium at the back. Laterally it fans out (forceps) into the hemispheres.
  {
    const spine = new THREE.CatmullRomCurve3([
      [-61, 3.4], [-56.5, 2.9], [-52, 3.6], [-50, 5.8], [-51.8, 8], [-58, 9.2],
      [-70, 9.7], [-82, 9.6], [-92, 9], [-98, 8.2], [-101, 6.6], [-99.5, 5.2],
    ].map(([z, y]) => V(0, y, z)), false, 'centripetal');
    const steps = mobile ? 120 : 220, around = mobile ? 28 : 48;
    const lateral = V(1, 0, 0), c = V(0, 0, 0), tangent = V(0, 0, 0), normal = V(0, 0, 0);
    const p = [], f = [], idx = [];
    for (let i = 0; i <= steps; i++) {
      const t = i / steps;
      spine.getPointAt(t, c);
      spine.getTangentAt(t, tangent);
      normal.crossVectors(tangent, lateral).normalize();
      const cap = tubeCaps(t, 14, 16);
      // Thin rostrum → thick genu → slimmer body → bulbous splenium.
      const thick = (.35 + .9 * bump(t, .2, .08) + .4 * smooth01(.12, .3, t) * (1 - smooth01(.8, .88, t)) + 1.5 * bump(t, .91, .06)) * cap;
      const width = (7 + 1.8 * bump(t, .2, .1) + 2.4 * bump(t, .92, .08)) * (.55 + .45 * cap);
      const cx = laneX(c.z), cy = laneY(c.z);
      for (let j = 0; j <= around; j++) {
        const a = (j / around) * Math.PI * 2;
        const side = Math.cos(a);
        const droop = side * side * 1.5 * cap;      // fibres sweep down into each hemisphere
        const v = Math.sin(a) * thick - droop;
        p.push(cx + side * width, cy + c.y + normal.y * v, c.z + normal.z * v);
        f.push(Math.sin(t * 120 + side * 1.5));     // transverse fibres crossing between hemispheres
        if (i < steps && j < around) {
          const k = i * (around + 1) + j, k2 = k + around + 1;
          idx.push(k, k2, k + 1, k2, k2 + 1, k + 1);
        }
      }
    }
    const callosum = new THREE.BufferGeometry();
    callosum.setAttribute('position', new THREE.Float32BufferAttribute(p, 3));
    callosum.setAttribute('aFold', new THREE.Float32BufferAttribute(f, 1));
    callosum.setIndex(idx);
    callosum.computeVertexNormals();
    group.add(new THREE.Mesh(callosum, makeTissueMaterial(THREE.DoubleSide, 1, 110)));
    addBrainLabel('CORPUS CALLOSUM', lanePoint(-56, 3, 12.5));
  }

  // Chapter 03 — hippocampus: a bulbous head with finger-like digitations,
  // a tapering body and tail, which rises and continues as the arching fornix.
  group.add(new THREE.Mesh(profiledTube([
    lanePoint(-104, -8, -5.2), lanePoint(-112, -9, -6), lanePoint(-122, -9.2, -6.2),
    lanePoint(-131, -8.4, -5.6), lanePoint(-138, -6.6, -3.4), lanePoint(-141.5, -4.8, -.4),
    lanePoint(-141, -3.6, 3), lanePoint(-136, -2.8, 5.6), lanePoint(-128, -2.2, 6.8), lanePoint(-120, -1.8, 7),
  ], (u, a) => {
    const head = 1 - smooth01(.15, .24, u);
    const body = mix(1.8, 1, smooth01(.24, .6, u));
    const fornix = mix(.55, .4, smooth01(.64, 1, u)) * (1 - .45 * Math.abs(Math.sin(a)));   // flat band
    let r = mix(mix(body, fornix, smooth01(.58, .68, u)), 2.7, head);
    const digit = Math.cos(u * Math.PI * 2 * 18);
    r *= 1 + .16 * head * Math.max(0, -digit);   // digitations of the hippocampal head
    return [r * tubeCaps(u, 9, 14), mix(3, digit, head)];
  }), makeTissueMaterial(THREE.FrontSide, 1, 110)));
  addBrainLabel('HIPPOCAMPUS', lanePoint(-112, -15, -1.5));

  // Chapter 04 — thalamus: paired ovoids with a narrow anterior tubercle, a bulky
  // pulvinar overhanging at the back, flat medial walls facing each other across
  // the third ventricle, and a Y-shaped internal medullary lamina on top.
  for (const side of [-1, 1]) {
    const thalamus = brainPart(mobile ? 56 : 96, mobile ? 40 : 72, 0, (d, out) => {
      const taper = d.z > 0 ? 1 - d.z * .35 : 1 + -d.z * .12;
      let x = d.x * 3.4 * taper, y = d.y * 3 * taper;
      const back = Math.max(0, -d.z);
      y -= back * back * Math.max(0, -d.y) * .7;                        // pulvinar overhang
      if (d.x * side > 0) x += side * d.x * side * back * .6;           // pulvinar bulges laterally
      if (d.x * side < 0) x *= .5;                                      // flat medial wall
      const lamina1 = d.x * side - .05 - .25 * Math.max(0, d.z);
      const lamina2 = d.x * side + .25 - .7 * Math.max(0, d.z - .25);
      const fold = d.y > 0 ? Math.min(Math.abs(lamina1), Math.abs(lamina2)) * 4 : 3;
      out.set(x, y, d.z * 6.2);
      return fold;
    });
    const mesh = new THREE.Mesh(thalamus, makeTissueMaterial(THREE.FrontSide, 1, 110));
    mesh.position.copy(lanePoint(-170, side * 6.4, -1.2));
    mesh.rotation.y = side * .12;
    group.add(mesh);
  }
  // Interthalamic adhesion: the small bridge between the two thalami (kept low, below the flight path).
  {
    const adhesion = brainPart(24, 16, 0, (d, out) => { out.set(d.x * 4.9, d.y * .55, d.z * .9); return 3; });
    const mesh = new THREE.Mesh(adhesion, makeTissueMaterial(THREE.FrontSide, .8, 110));
    mesh.position.copy(lanePoint(-166, 0, -3.4));
    group.add(mesh);
  }
  addBrainLabel('THALAMUS', lanePoint(-160, 4.5, 5.2));

  // Chapter 05 — caudate nucleus: a bulbous head, a flattened body arching back and
  // a thin tail curling down and forward, ending at the almond-shaped amygdala.
  group.add(new THREE.Mesh(profiledTube([
    lanePoint(-204, 6.5, 2.5), lanePoint(-210, 7.5, 4.5), lanePoint(-218, 8.6, 6.2),
    lanePoint(-227, 9.4, 6.4), lanePoint(-235, 9.8, 4.8), lanePoint(-239, 9.8, 1.5),
    lanePoint(-237, 9.5, -2.2), lanePoint(-231, 9, -4.8), lanePoint(-223, 8.6, -6),
  ], (u, a) => {
    let r = mix(3.3, 1.5, smooth01(.04, .3, u));
    r = mix(r, .35, smooth01(.45, .95, u));
    r *= 1 - .25 * Math.abs(Math.sin(a));        // comma-shaped section hugging the ventricle
    return [r * tubeCaps(u, 7, 14), 3];
  }), makeTissueMaterial(THREE.FrontSide, 1, 110)));
  {
    const amygdala = brainPart(40, 30, 0, (d, out) => { out.set(d.x * 1.6, d.y * 1.4, d.z * 2.1); return 3; });
    const mesh = new THREE.Mesh(amygdala, makeTissueMaterial(THREE.FrontSide, 1, 110));
    mesh.position.copy(lanePoint(-220.5, 8.4, -6.4));
    group.add(mesh);
  }
  addBrainLabel('CAUDATE NUCLEUS', lanePoint(-208, 12.5, 6.5));

  buildVagusJourney(group);
  scene.add(group);
}

// =====================================
// FINALE — brainstem → vagus nerve → heart, lungs and gut
// =====================================
// The vagus (cranial nerve X) leaves the medulla oblongata as a row of rootlets,
// descends the neck beside the trachea and oesophagus, and branches to the heart,
// the lungs, the stomach and the intestines. Signals visibly travel down it.
let heartMesh = null;
const lungMeshes = [];

function makeNerveMaterial() {
  const material = new THREE.ShaderMaterial({
    uniforms: {
      uTime: brainMaterial.uniforms.uTime,
      uNetwork: networkFade,
      uPale: brainMaterial.uniforms.uPale,
      uSignal: {value: new THREE.Color('#44eaff')},
    },
    transparent: true, depthWrite: false, depthTest: false,
    blending: THREE.AdditiveBlending, side: THREE.FrontSide,
    vertexShader: `
      varying float vU;
      varying float vRim;
      varying float vDepth;
      void main() {
        vU = uv.x;
        vec4 mv = modelViewMatrix * vec4(position, 1.0);
        vec3 n = normalize(normalMatrix * normal);
        vRim = 1.0 - abs(dot(n, normalize(-mv.xyz)));
        vDepth = -mv.z;
        gl_Position = projectionMatrix * mv;
      }`,
    fragmentShader: `
      uniform float uTime;
      uniform float uNetwork;
      uniform vec3 uPale;
      uniform vec3 uSignal;
      varying float vU;
      varying float vRim;
      varying float vDepth;
      void main() {
        float fog = (1.0 - smoothstep(50.0, 150.0, vDepth)) * smoothstep(.8, 4.0, vDepth);   // no glare right at the lens
        // Impulses running from the brainstem down to the organs.
        float q = fract(vU * 4.0 - uTime * .45) - .5;
        float pulse = exp(-q * q * 160.0);
        vec3 col = uPale * (.07 + vRim * .26) + uSignal * pulse * (.3 + vRim * .35);
        gl_FragColor = vec4(col * fog * uNetwork * .8, 1.0);
      }`,
  });
  interiorMaterials.push(material);
  return material;
}

// Thin nerve tube: radius tapers r0 → r1, uv.x runs from origin to target.
function nerveTube(points, r0, r1, segments = 70) {
  const curve = new THREE.CatmullRomCurve3(points);
  const radial = mobile ? 6 : 10;
  const geometry = new THREE.TubeGeometry(curve, segments, 1, radial, false);
  const position = geometry.attributes.position;
  const uv = geometry.attributes.uv;
  const centre = V(0, 0, 0), offset = V(0, 0, 0);
  for (let i = 0; i < position.count; i++) {
    const u = uv.getX(i);
    curve.getPointAt(u, centre);
    offset.fromBufferAttribute(position, i).sub(centre).multiplyScalar(mix(r0, r1, u) * tubeCaps(u, 30, 20)).add(centre);
    position.setXYZ(i, offset.x, offset.y, offset.z);
  }
  geometry.computeVertexNormals();
  return geometry;
}

function buildVagusJourney(group) {
  const organMaterial = () => makeTissueMaterial(THREE.FrontSide, 1, 120);
  const L = lanePoint;   // L(z, sideways, up) relative to the flight path

  // Brainstem below the road: midbrain → bulging pons → medulla oblongata → spinal cord.
  group.add(new THREE.Mesh(profiledTube([
    L(-250, 0, -4.2), L(-256, 0, -4.8), L(-262, 0, -5.6), L(-268, 0, -6.4), L(-276, 0, -7.6), L(-290, 0, -9.5),
  ], (u) => {
    const r = 2 + 1.1 * bump(u, .3, .1) - .3 * smooth01(.4, .55, u) - .6 * smooth01(.6, .8, u);
    const fold = mix(3, Math.cos(u * Math.PI * 2 * 26), smooth01(.25, .7, bump(u, .3, .1)));
    return [r * tubeCaps(u, 10, 30), fold];
  }), organMaterial()));
  addBrainLabel('BRAINSTEM', L(-252, 4, -1.6));

  // One connected vagus system: rootlets → two trunks running along the road
  // beside the oesophagus → branches to heart, lungs, stomach and intestines.
  const nerves = [];
  for (const s of [-1, 1]) {
    for (let k = 0; k < 6; k++) {
      nerves.push(nerveTube([
        L(-263.5 - k * .9, s * 1.5, -5.5 - k * .1),
        L(-266.5 - k * .5, s * 2.3, -5.1),
        L(-270.5, s * 2.6, -4.6),
      ], .06, .09, 24));
    }
    const trunk = s < 0
      // Left (anterior) vagal trunk ends on the stomach.
      ? [L(-270.5, -2.6, -4.6), L(-280, -2.4, -3.6), L(-292, -2, -3), L(-305, -1.8, -2.6), L(-318, -1.4, -2.4),
        L(-330, -1.2, -2.4), L(-340, -1.2, -2.6), L(-348, -2.2, -3), L(-353, -3.4, -3.2)]
      // Right (posterior) vagal trunk continues to the coeliac plexus and the gut.
      : [L(-270.5, 2.6, -4.6), L(-280, 2.4, -3.6), L(-292, 2, -3), L(-305, 1.8, -2.6), L(-318, 1.4, -2.4),
        L(-330, 1.2, -2.4), L(-340, 1.2, -2.8), L(-350, 1.4, -3.8), L(-360, .8, -5.2), L(-367, 0, -6)];
    nerves.push(nerveTube(trunk, .2, .15, 260));
    // Cardiac branches.
    nerves.push(nerveTube([L(-303, s * 1.8, -2.6), L(-309, s * .9 - 2.4, -3.2), L(-314, -4.4 + s * .8, -2)], .1, .07, 40));
    // Pulmonary branches into each lung's hilum.
    for (const dz of [0, 2.5]) {
      nerves.push(nerveTube([L(-312 - dz, s * 1.5, -2.4), L(-312.5 - dz, s * 4.5, -.8), L(-313 - dz, s * 7.4, .2)], .1, .06, 30));
    }
  }
  // Left recurrent laryngeal nerve hooks under the aortic arch and climbs back up along the trachea.
  nerves.push(nerveTube([
    L(-306, -1.9, -2.6), L(-309, -2.8, -1), L(-308, -2.6, 1.2), L(-302, -1.4, 2), L(-292, -1, 2.2), L(-282, -.9, 2.2),
  ], .09, .06, 60));
  // Gastric branches spreading over the stomach.
  for (const [z, x, y] of [[-357, -7.2, -2.6], [-361, -7.6, -5.4], [-366, -4.4, -6.8]]) {
    nerves.push(nerveTube([L(-353, -3.4, -3.2), L((-353 + z) / 2, (-3.4 + x) / 2, (-3.2 + y) / 2 + .6), L(z, x, y)], .09, .05, 30));
  }
  // Coeliac branches to the intestines.
  for (const [z, x, y] of [[-376, -3, -8], [-383, 3.5, -7.5], [-390, -1, -9]]) {
    nerves.push(nerveTube([L(-367, 0, -6), L((-367 + z) / 2, x * .5, (-6 + y) / 2 + .5), L(z, x, y)], .09, .05, 40));
  }
  group.add(new THREE.Mesh(mergeGeometries(nerves), makeNerveMaterial()));
  addBrainLabel('VAGUS NERVE', L(-276, 3.5, -1.4));

  // Trachea overhead with cartilage rings, splitting into the two main bronchi.
  group.add(new THREE.Mesh(profiledTube([L(-272, 0, 2.6), L(-284, 0, 3), L(-296, 0, 3.2), L(-300, 0, 3.2)],
    (u) => [1.1 * tubeCaps(u, 30, 60), Math.cos(u * Math.PI * 2 * 22)], 20), organMaterial()));
  for (const s of [-1, 1]) {
    group.add(new THREE.Mesh(profiledTube([L(-300, 0, 3.2), L(-304, s * 4.5, 2.2), L(-308, s * 7.5, 1.2)],
      (u) => [mix(.8, .55, u) * tubeCaps(u, 30, 12), Math.cos(u * Math.PI * 2 * 7)], 16), organMaterial()));
  }
  addBrainLabel('TRACHEA', L(-282, 2, 5.2));

  // Lungs on both sides of the road: narrow apex, broad base, concave medial surface,
  // an oblique fissure (plus a horizontal fissure on the right lung). They breathe.
  for (const s of [-1, 1]) {
    const lung = brainPart(mobile ? 48 : 80, mobile ? 40 : 64, 0, (d, out) => {
      let x = d.x * 4.4, y = d.y * 7.5, z = d.z * 7;
      if (d.y > 0) { x *= 1 - d.y * .4; z *= 1 - d.y * .3; }
      if (d.y < 0) y *= .82;
      if (d.x * s < 0) x *= .5;
      const oblique = d.y + d.z * .6 - .1;
      const horizontal = d.y - .18 + (d.z < 0 ? 9 : 0);
      const fold = (s > 0 ? Math.min(Math.abs(oblique), Math.abs(horizontal)) : Math.abs(oblique)) * 3;
      out.set(x, y, z);
      return fold;
    });
    const mesh = new THREE.Mesh(lung, organMaterial());
    mesh.position.copy(L(-313, s * 10, .5));
    group.add(mesh);
    lungMeshes.push(mesh);
  }
  // Beside the right lung, at the height of its centre.
  addBrainLabel('LUNGS', L(-306, 5.8, 1));

  // Heart between the lungs, low on the left: apex down and to the left, coronary and
  // interventricular grooves, beating; the aortic arch rises from it and turns back.
  const heart = brainPart(mobile ? 48 : 80, mobile ? 36 : 60, 0, (d, out) => {
    let x = d.x * 2.8, y = d.y * 3.2, z = d.z * 2.5;
    if (d.y < 0) { const k = 1 + d.y * .55; x *= k; z *= k; }
    const coronary = d.y - .25 + d.x * .2;
    const interventricular = d.x * .9 + d.z * .4 - .05;
    out.set(x, y, z);
    return Math.min(Math.abs(coronary), Math.abs(interventricular) + (d.y > .3 ? 9 : 0)) * 3;
  });
  heartMesh = new THREE.Mesh(heart, organMaterial());
  heartMesh.position.copy(L(-317, -5, -4));
  heartMesh.rotation.set(.2, .3, .45);
  group.add(heartMesh);
  group.add(new THREE.Mesh(profiledTube([
    L(-315, -4.4, -1.2), L(-311, -3.6, 1), L(-308, -2.4, 1.6), L(-305.5, -1.4, .4), L(-308, -1.3, -3.6), L(-322, -1.4, -4.8), L(-338, -1, -5.4),
  ], (u) => [.8 * tubeCaps(u, 14, 14), 3], 18), organMaterial()));
  // Right next to the heart, on the left side of the road.
  addBrainLabel('HEART', L(-313, -2, -3.6));

  // Oesophagus along the floor of the road, down through the diaphragm to the stomach.
  group.add(new THREE.Mesh(profiledTube([L(-272, 0, -1.6), L(-300, 0, -2.4), L(-330, 0, -2.4), L(-342, -1, -2.8), L(-350, -3, -3.4)],
    (u) => [.8 * tubeCaps(u, 30, 30), 3], 16), organMaterial()));

  // Diaphragm: a muscular dome across the road that we fly through.
  {
    const rings = mobile ? 24 : 40, seg = mobile ? 56 : 96;
    const p = [], f = [], idx = [];
    for (let i = 0; i <= rings; i++) {
      const r = i / rings;
      for (let j = 0; j <= seg; j++) {
        const a = (j / seg) * Math.PI * 2;
        const dome = (1 - r * r) * 3.6 + Math.max(0, Math.cos(a)) * (1 - r) * .6;   // right dome sits a little higher
        p.push(laneX(-340) + Math.cos(a) * r * 15, laneY(-340) + Math.sin(a) * r * 11 - 1.5, -340 + dome);
        f.push(Math.sin(a * 26) * 1.2 + (r < .35 ? 3 : 0));                             // radiating muscle fibres, central tendon smooth
        if (i < rings && j < seg) {
          const k = i * (seg + 1) + j, k2 = k + seg + 1;
          idx.push(k, k2, k + 1, k2, k2 + 1, k + 1);
        }
      }
    }
    const diaphragm = new THREE.BufferGeometry();
    diaphragm.setAttribute('position', new THREE.Float32BufferAttribute(p, 3));
    diaphragm.setAttribute('aFold', new THREE.Float32BufferAttribute(f, 1));
    diaphragm.setIndex(idx);
    diaphragm.computeVertexNormals();
    group.add(new THREE.Mesh(diaphragm, makeTissueMaterial(THREE.DoubleSide, .6, 120)));
    addBrainLabel('DIAPHRAGM', L(-334, 5, 5.5));
  }

  // Stomach on the left beyond the diaphragm: cardia, domed fundus, body with rugae,
  // antrum, pylorus and the first part of the duodenum.
  group.add(new THREE.Mesh(profiledTube([
    L(-350, -3, -3.4), L(-352, -6.5, -.8), L(-357, -8, -3.5), L(-363, -6.5, -6.5),
    L(-367, -3, -7), L(-368, .5, -5.5), L(-370, 2.5, -6.5), L(-373, 3, -8.5),
  ], (u, a) => {
    const r = 1 + 2.2 * bump(u, .15, .1) + 1.8 * smooth01(.1, .3, u) * (1 - smooth01(.55, .72, u)) + .8 * bump(u, .6, .08)
      - .4 * bump(u, .78, .04);
    const rugae = smooth01(.12, .3, u) * (1 - smooth01(.6, .7, u));
    return [r * tubeCaps(u, 30, 14), mix(3, Math.sin(a * 5 + u * 12), rugae)];
  }), organMaterial()));
  addBrainLabel('STOMACH', L(-352, -1, 1.8));

  // Small intestine: loops filling the space below the end of the road.
  const coil = [];
  for (let t = 0; t <= 6.2; t += .12) {
    coil.push(L(-376 - t * 3.4 - Math.sin(t * 4.7) * 1.6, Math.sin(t * 3.1) * 5.5, -7 - Math.cos(t * 2.3) * 2.6));
  }
  group.add(new THREE.Mesh(profiledTube(coil,
    (u) => [.85 * tubeCaps(u, 60, 60), Math.cos(u * Math.PI * 2 * 80)], 14), organMaterial()));
  addBrainLabel('INTESTINES', L(-374, 4.5, -2.5));
}

const _labelDistance = V(0, 0, 0);
function updateInterior() {
  for (const material of interiorMaterials) material.uniforms.uTime.value = internalTime;
  // Heartbeat ("lub-dub") and slow breathing.
  if (heartMesh) {
    const phase = (internalTime * 1.1) % 1;
    const beat = Math.exp(-(((phase * 10) - 1) ** 2)) + .6 * Math.exp(-(((phase * 10) - 2.6) ** 2));
    heartMesh.scale.setScalar(1 + beat * .045);
    const breath = 1 + Math.sin(internalTime * .9) * .03;
    for (const lung of lungMeshes) lung.scale.set(breath, 1 + (breath - 1) * 1.4, breath);
  }
  for (const label of brainLabels) {
    const d = _labelDistance.subVectors(label.position, camera.position).length();
    const ahead = label.position.z < camera.position.z ? 1 : 0;
    label.material.opacity = (1 - smooth01(label.userData.far * .55, label.userData.far, d)) * smooth01(7, 14, d) * ahead * networkFade.value * .85;
    label.visible = label.material.opacity > .005;
  }
}

// =====================================
// WATER-LIKE CURSOR / IRIDESCENT WAKE
// =====================================
const ink = inkCanvas.getContext('2d', {alpha:true});
const trails=[];
const ripples=[];
let lastPointer=null;
let orbX=-100,orbY=-100;
let activePointer=false;
let lastTrailAt=0;
const INK_LIFE=1.5;   // seconds a soft smudge takes to dissolve

function handlePointer(event) {
  if (event.pointerType === 'touch' || reducedMotion) return;
  const rect=stage.getBoundingClientRect();
  const x=event.clientX-rect.left;
  const y=event.clientY-rect.top;
  targetPointerX=clamp((x/width-.5)*2,-1,1);
  targetPointerY=clamp((y/height-.5)*2,-1,1);
  activePointer=true;
  if(lastPointer) {
    const dx=x-lastPointer.x;
    const dy=y-lastPointer.y;
    const d=Math.hypot(dx,dy);
    if(d>1.8 && performance.now()-lastTrailAt>12) {
      // Interpolate the wake for fast mouse movement; this makes it feel like a brush.
      const steps=Math.min(8,Math.ceil(d/16));
      for(let s=1;s<=steps;s++) {
        const t=s/steps;
        trails.push({
          x:mix(lastPointer.x,x,t), y:mix(lastPointer.y,y,t),
          age:0, vx:dx*.012+rand(-.25,.25), vy:dy*.01+rand(-.25,.25),
          hue:(190+rand(-30,70))%360,
          radius:rand(1.6,3.4),
        });
      }
      lastTrailAt=performance.now();
    }
  }
  lastPointer={x,y};
}
stage.addEventListener('pointermove',handlePointer,{passive:true});
stage.addEventListener('pointerleave',()=>{
  activePointer=false;
  lastPointer=null;
  targetPointerX=targetPointerY=0;
  cursorOrb.style.opacity='0';
},{passive:true});
stage.addEventListener('pointerenter',()=>{if(!reducedMotion)cursorOrb.style.opacity='1';});
stage.addEventListener('pointerdown',(e)=>{
  if(reducedMotion || e.pointerType==='touch')return;
  const rect=stage.getBoundingClientRect();
  ripples.push({x:e.clientX-rect.left,y:e.clientY-rect.top,age:0});
});

// Chapter 05: drag (mouse or finger) to spin the brain a full 360°, with inertia.
let brainReveal=0;
let brainSpin=0;
let brainSpinVelocity=0;
let brainTilt=0;
let brainDrag=null;
stage.addEventListener('pointerdown',(e)=>{
  if(brainReveal<.3 || e.target.closest('a,button'))return;
  brainDrag={id:e.pointerId,x:e.clientX,y:e.clientY,t:performance.now()};
  brainSpinVelocity=0;
  stage.classList.add('is-dragging');
});
stage.addEventListener('pointermove',(e)=>{
  if(!brainDrag || e.pointerId!==brainDrag.id)return;
  const now=performance.now();
  const dx=e.clientX-brainDrag.x;
  const dy=e.clientY-brainDrag.y;
  const delta=dx*.009;
  brainSpin+=delta;
  brainTilt=clamp(brainTilt+dy*.004,-.35,.35);
  brainSpinVelocity=mix(brainSpinVelocity,delta/Math.max(.008,(now-brainDrag.t)/1000),.5);
  brainDrag.x=e.clientX;
  brainDrag.y=e.clientY;
  brainDrag.t=now;
},{passive:true});
const endBrainDrag=()=>{brainDrag=null;stage.classList.remove('is-dragging');};
window.addEventListener('pointerup',endBrainDrag,{passive:true});
window.addEventListener('pointercancel',endBrainDrag,{passive:true});

function animateCursor(dt) {
  ink.clearRect(0,0,width,height);
  if(reducedMotion) return;
  if(activePointer && lastPointer) {
    const follow=1-Math.exp(-dt*14);
    orbX=mix(orbX<0?lastPointer.x:orbX,lastPointer.x,follow);
    orbY=mix(orbY<0?lastPointer.y:orbY,lastPointer.y,follow);
    cursorOrb.style.transform=`translate3d(${orbX-10.5}px,${orbY-10.5}px,0)`;
  }
  // Soft smudges drift and spread slowly, like light dissolving on water.
  for(let i=trails.length-1;i>=0;i--) {
    const p=trails[i];
    p.age+=dt;
    p.vx*=Math.exp(-dt*1.8);
    p.vy*=Math.exp(-dt*1.8);
    p.x+=p.vx*dt*30 + Math.sin(p.age*4+p.hue)*dt*2;
    p.y+=p.vy*dt*30 + Math.cos(p.age*3.5+p.hue)*dt*2;
    if(p.age>INK_LIFE) trails.splice(i,1);
  }
  ink.save();
  ink.globalCompositeOperation='lighter';
  // Wide, faint, colour-blended glows — no bright cores.
  for(let i=0;i<trails.length;i++) {
    const p=trails[i];
    const life=Math.pow(1-p.age/INK_LIFE,2);
    if(life<=0)continue;
    const r=p.radius*(3+(1-life)*6)*3;   // grows softer as it fades
    const grad=ink.createRadialGradient(p.x,p.y,0,p.x,p.y,r);
    grad.addColorStop(0,`hsla(${p.hue},90%,68%,${life*.13})`);
    grad.addColorStop(.4,`hsla(${(p.hue+45)%360},90%,60%,${life*.05})`);
    grad.addColorStop(1,`hsla(${p.hue},90%,55%,0)`);
    ink.fillStyle=grad;
    ink.beginPath();
    ink.arc(p.x,p.y,r,0,Math.PI*2);
    ink.fill();
  }
  // A whisper-thin, silky line through the freshest points.
  const recent=trails.filter(p=>p.age<.45);
  if(recent.length>2) {
    ink.lineCap='round';
    ink.lineJoin='round';
    for(let pass=0;pass<2;pass++) {
      ink.lineWidth=pass===0?14:1.4;
      ink.strokeStyle=pass===0?'rgba(171,86,255,.03)':'rgba(136,247,255,.09)';
      ink.beginPath();
      ink.moveTo(recent[0].x,recent[0].y);
      for(let i=1;i<recent.length-1;i++) {
        const p=recent[i],q=recent[i+1];
        ink.quadraticCurveTo(p.x,p.y,(p.x+q.x)*.5,(p.y+q.y)*.5);
      }
      ink.stroke();
    }
  }
  for(let i=ripples.length-1;i>=0;i--) {
    const r=ripples[i];r.age+=dt;
    const alpha=Math.pow(Math.max(0,1-r.age/1.1),2);
    if(r.age>1.1){ripples.splice(i,1);continue;}
    ink.lineWidth=1;
    ink.strokeStyle=`rgba(115,233,255,${alpha*.22})`;
    ink.beginPath();ink.arc(r.x,r.y,12+r.age*110,0,Math.PI*2);ink.stroke();
    ink.strokeStyle=`rgba(171,120,255,${alpha*.14})`;
    ink.beginPath();ink.arc(r.x,r.y,4+r.age*68,0,Math.PI*2);ink.stroke();
  }
  ink.restore();
}

// =====================================
// SCROLL-DRIVEN 3D CAMERA + UI
// =====================================
function readScroll() {
  const distance=Math.max(1,journey.offsetHeight-window.innerHeight);
  const offset=window.scrollY + journey.getBoundingClientRect().top; // absolute top
  targetProgress=clamp((window.scrollY-offset)/distance,0,1);
}
function switchChapter(index) {
  if(index===chapterIndex)return;
  chapterIndex=index;
  chapters.forEach((c,i)=>{
    c.classList.toggle('is-active',i===index);
    c.setAttribute('aria-hidden',String(i!==index));
    c.querySelectorAll('a,button').forEach(el => { el.tabIndex = i === index ? 0 : -1; });
  });
  jumpButtons.forEach((b,i)=>{
    b.classList.toggle('is-current',i===index);
    if(i===index)b.setAttribute('aria-current','step');
    else b.removeAttribute('aria-current');
  });
}
function updateHud(progress) {
  const number=Math.round(progress*100);
  progressFill.style.width=`${number}%`;
  progressLabel.textContent=`${String(number).padStart(2,'0')}%`;
  const depth=progress*CONFIG.journeyDistance;
  depthLabel.textContent=`DEPTH ${depth.toFixed(1).padStart(5,'0')}`;
  let chapter=0;
  for(const boundary of CHAPTER_DEPTHS) if(depth>=boundary) chapter++;
  switchChapter(chapter);
}
function moveToChapter(index) {
  const distance=Math.max(1,journey.offsetHeight-window.innerHeight);
  const destinations=CHAPTER_JUMPS.map(depth=>depth/CONFIG.journeyDistance);
  const absoluteY=window.scrollY+journey.getBoundingClientRect().top;
  window.scrollTo({top:absoluteY+distance*destinations[index],behavior:reducedMotion?'auto':'smooth'});
}
jumpButtons.forEach(btn=>btn.addEventListener('click',()=>moveToChapter(Number(btn.dataset.jump))));
function setMotion(enabled){
  reducedMotion=!enabled;
  motionButton.setAttribute('aria-pressed',String(reducedMotion));
  motionLabel.textContent=enabled?'MOTION ON':'MOTION OFF';
  if(reducedMotion){
    trails.length=0;ripples.length=0;
    ink.clearRect(0,0,inkCanvas.width,inkCanvas.height);
    cursorOrb.style.opacity='0';
  }
}
motionButton.addEventListener('click',()=>setMotion(reducedMotion));
setMotion(!reducedMotion);
prefersReducedMotion.addEventListener?.('change',(event)=>setMotion(!event.matches));
window.addEventListener('scroll',readScroll,{passive:true});

function resize() {
  width=Math.max(1,window.innerWidth);
  height=Math.max(1,window.innerHeight);
  dpr=Math.min(window.devicePixelRatio||1,mobile?CONFIG.maxPixelRatioMobile:CONFIG.maxPixelRatioDesktop);
  renderer.setPixelRatio(dpr);
  renderer.setSize(width,height,false);
  composer.setPixelRatio(dpr);
  composer.setSize(width,height);
  camera.aspect=width/height;
  camera.fov=width<760?69:65;
  camera.updateProjectionMatrix();
  inkCanvas.width=Math.round(width);
  inkCanvas.height=Math.round(height);
  inkCanvas.style.width=width+'px';
  inkCanvas.style.height=height+'px';
  particleMaterial.uniforms.uPixelRatio.value=dpr;
  readScroll();
}
// Critically damped spring: the camera eases in, glides and brakes without overshoot.
const progressSpring={v:0};
function smoothDamp(current,target,state,smoothTime,dt){
  const omega=2/smoothTime;
  const x=omega*dt;
  const exp=1/(1+x+.48*x*x+.235*x*x*x);
  const change=current-target;
  const temp=(state.v+omega*change)*dt;
  state.v=(state.v-omega*temp)*exp;
  let out=target+(change+temp)*exp;
  if((target-current>0)===(out>target)){out=target;state.v=0;}
  return out;
}
const lookAhead=V(0,0,0);
let bank=0;
function animate(){
  requestAnimationFrame(animate);
  const dt=Math.min(clock.getDelta(),.06);
  smoothProgress=smoothDamp(smoothProgress,targetProgress,progressSpring,reducedMotion?.1:.7,dt);
  const pointerDamping=1-Math.exp(-dt*3.2);
  pointerX=mix(pointerX,targetPointerX,pointerDamping);
  pointerY=mix(pointerY,targetPointerY,pointerDamping);
  if(!reducedMotion)internalTime+=dt;
  // The camera rides exactly on the curved lane and looks further along it,
  // so close neurons sweep past on every side as it flies forward.
  const z=18-smoothProgress*CONFIG.journeyDistance;
  const sway=reducedMotion?0:1;
  cameraSmooth.set(
    laneX(z)+pointerX*(mobile?.1:.5)+Math.sin(internalTime*.37)*.1*sway,
    laneY(z)-pointerY*(mobile?.05:.32)+Math.sin(internalTime*.5)*.08*sway,
    z,
  );
  lookAhead.set(laneX(z-16),laneY(z-16),z-16);
  camera.position.copy(cameraSmooth);
  camera.lookAt(lookAhead);
  camera.rotateY(-pointerX*.06);
  camera.rotateX(-pointerY*.04);
  // Bank softly into the curves of the lane.
  const turn=laneX(z-10)-laneX(z);
  bank=mix(bank,clamp(turn*.05,-.08,.08),1-Math.exp(-dt*2));
  camera.rotateZ(bank+pointerX*.01);
  lineMaterial.uniforms.uTime.value=internalTime;
  lineMaterial.uniforms.uProgress.value=smoothProgress;
  particleMaterial.uniforms.uTime.value=internalTime;
  particleMaterial.uniforms.uProgress.value=smoothProgress;
  for(const material of auroraMaterials){
    material.uniforms.uTime.value=internalTime;
    material.uniforms.uProgress.value=smoothProgress;
  }
  // Chapter 01 intro: the brain materialises on load; the first scroll flies the
  // camera into it, the brain dissolves around us and the network lights up inside.
  introTime+=dt;
  const intro=reducedMotion?1:smooth01(.3,2.6,introTime);
  const flown=smoothProgress*CONFIG.journeyDistance;   // depth travelled
  const inside=smooth01(13.4,26.8,flown);
  const fade=1-inside;
  brainReveal=intro*fade;
  brainGroup.visible=fade>.001;
  brainMaterial.uniforms.uReveal.value=intro;
  brainMaterial.uniforms.uFade.value=fade;
  brainMaterial.uniforms.uTime.value=internalTime;
  stage.classList.toggle('is-brain',brainReveal>.3);
  networkFade.value=smooth01(6.7,21.4,flown);
  networkDim.value=networkFade.value*(1-.6*smooth01(262,290,flown));
  for(const material of fadeMaterials) material.opacity=material.userData.baseOpacity*networkFade.value;
  updateInterior();
  // Released spin glides on and slowly comes to rest.
  if(!brainDrag){
    brainSpin+=brainSpinVelocity*dt;
    brainSpinVelocity*=Math.exp(-dt*1.8);
    brainTilt*=Math.exp(-dt*.8);
  }
  // Cursor left/right turns the brain; drag spins it fully. Classic lateral profile at rest.
  const idle=reducedMotion?0:1;
  brainGroup.rotation.y=-.22+brainSpin+pointerX*.9*fade+Math.sin(internalTime*.12)*.1*idle;
  brainGroup.rotation.x=.1+brainTilt+pointerY*.18*fade+Math.sin(internalTime*.09)*.03*idle;
  if(!document.hidden){
    composer.render(dt);
    animateCursor(dt);
  }
  updateHud(smoothProgress);
}

// =====================================
// START
// =====================================
try {
  if(!ink) throw new Error('Your browser does not support Canvas 2D.');
  makeRenderer();
  // Aurora veils disabled: flying through them washed the whole screen in a pink glassy haze.
  const nodes=buildNeuralFibres();
  buildParticles(nodes);
  buildPortals();
  buildBrain();
  buildInterior();
  resize();
  window.addEventListener('resize',resize,{passive:true});
  window.addEventListener('orientationchange',resize,{passive:true});
  readScroll();
  smoothProgress=targetProgress;
  loadingText.textContent='Połączenia gotowe — rozpocznij podróż.';
  animate();
  // Remove loader after two frames, when the WebGL output is ready.
  requestAnimationFrame(()=>requestAnimationFrame(()=>loading.classList.add('is-ready')));
} catch(error) {
  console.error('Neural Journey:',error);
  loadingText.textContent='Nie udało się uruchomić WebGL. Sprawdź obsługę grafiki 3D i konsolę przeglądarki.';
  $('.loading__bar').style.display='none';
}
