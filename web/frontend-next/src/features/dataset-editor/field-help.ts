export type DatasetHelpContent = {
  summary: string;
  recommendation: string;
  note?: string;
  options?: readonly { value: string; label: string; when: string }[];
  advanced?: readonly string[];
};

export const DATASET_FIELD_HELP = {
  resolution: {
    summary: '设置训练图片的目标尺寸，例如 1024 表示以 1024 × 1024 的面积预算选择尺寸。',
    recommendation: '先沿用蓝图的分辨率，再根据模型要求、显存和图片细节调整。分辨率越高，通常越费显存和时间。',
    note: '分桶开启时可以使用长方形尺寸，处理过程仍可能裁剪边缘。关闭分桶时统一缩放、裁剪到这个正方形尺寸。',
  },
  batch_size: {
    summary: '一次训练计算从这个数据集组取出的图片数量。',
    recommendation: '显存紧张时先用 1，想增加有效批量可在训练配置中调整梯度累积。',
    note: 'Qwen 图像编辑的目标子集必须设为 1。',
  },
  prior_loss_weight: {
    summary: '控制正则化图片的损失相对普通图片占多大权重。',
    recommendation: '使用正则子集时先保留 1，表示两类图片采用相同倍率。调高会加强正则图片对训练的影响。',
    note: '你需要同时勾选子集的“正则数据”，单独设置权重不会改变子集类型。',
  },
  enable_bucket: {
    summary: '将不同比例的图片分到接近的训练尺寸，减少为了统一正方形而裁掉的内容。',
    recommendation: '竖图、横图混用时保持开启。只有你希望所有图片都缩放、裁剪成同一个正方形尺寸时才关闭。',
    note: '分桶后仍会等比缩放并裁剪到桶尺寸，不保证保留原图全部内容。',
  },
  min_bucket_reso: {
    summary: '标准分桶允许的最小边长，例如 256 表示不生成边长小于 256 的桶。',
    recommendation: '先沿用蓝图值，只有需要更窄的长图桶时再调低。不要为了小图一味降低训练尺寸。',
    note: '启用固定 token 分桶且训练分辨率为 1024 时使用固定尺寸表，本项不筛选该表。',
    advanced: ['固定 token 分桶让每个桶包含固定数量的图像片段。其他分辨率会缩放这张尺寸表，再按最小、最大边长筛选。'],
  },
  max_bucket_reso: {
    summary: '标准分桶允许的最大边长，影响横图或竖图可用的最长一边。',
    recommendation: '先沿用蓝图值，需要更长的横图或竖图桶时再增加。它与训练分辨率的面积预算一起决定桶形状。',
    note: '启用固定 token 分桶且训练分辨率为 1024 时使用固定尺寸表，本项不限制该表的最长边。',
    advanced: ['其他分辨率的固定 token 桶会先缩放，再按最小、最大边长筛选。固定尺寸表的最长边可能大于训练分辨率。'],
  },
  bucket_reso_steps: {
    summary: '标准分桶尺寸的间隔，例如 64 表示按 64 像素递增。',
    recommendation: '先保持 64，只有需要更细的桶尺寸时再减小。更小的间隔通常会增加桶数量。',
    note: '固定 token 桶表不按这个间隔生成。禁止放大且找不到合适桶时，预处理仍会用它对齐后备尺寸。',
  },
  bucket_no_upscale: {
    summary: '预处理时优先选择不超过原图尺寸的桶，减少小图被放大。',
    recommendation: '如果你更在意保留小图原有清晰度，可以开启。关闭分桶时这一项不生效。',
    note: '极小图片没有合适桶时仍受最小桶尺寸限制，不能保证所有图片都完全不放大。',
  },
  validation_split: {
    summary: '从普通训练图片中按比例留出验证集，例如 0.1 表示约 10%。',
    recommendation: '需要跟踪验证表现时再设置比例，不需要时保持 0。“验证集数量”大于 0 时优先按数量切分。',
    note: '同一数据集组的非正则训练图片，应用递归与路径筛选后合计为 1～99 张时，会自动关闭验证切分。重复次数不计入这个门槛。',
    advanced: ['门槛按实际 image_dir 中的图片计算，在验证切分和 sample_ratio 抽样之前计数，不是按原始目录或整份蓝图的总张数判断。'],
  },
  validation_split_num: {
    summary: '指定留出的验证图片数量，优先级高于“验证集比例”。',
    recommendation: '需要固定数量时填写正整数。设为 0 才会使用比例，两项都为 0 时不切验证集。',
    note: '同一数据集组的非正则训练图片，应用递归与路径筛选后合计为 1～99 张时，会自动关闭验证切分。重复次数不计入这个门槛。',
    advanced: ['判断使用实际训练图片目录 image_dir，在切分和 sample_ratio 抽样之前计数。'],
  },
  validation_seed: {
    summary: '控制验证图片抽样的随机顺序。',
    recommendation: '比较不同训练设置时固定这个整数。只有你想换一批验证图片时再改。',
    note: '图片内容、数量或筛选规则改变后，即使种子相同，也不保证仍是原来的验证图片。',
  },
  caption_extension: {
    summary: '告诉训练器去哪个同名文本文件找图片标注，例如 cat.png 对应 cat.txt。',
    recommendation: '保持 .txt。只有你的文本标注使用其他后缀（例如 .caption）时才改。',
    note: '使用 JSON 标注来源时不靠这一项选择 JSON 文件。后缀填错会读不到文本标注。',
    advanced: ['标注来源设为 auto 时，先查 captions.json 和同名 JSON，最后才尝试这个后缀。预处理、缓存和预览支持这些来源，无缓存训练直接读取文本时仍使用本项。'],
  },
  keep_tokens: {
    summary: '在标签丢弃或逐步加入标签时，保护标注开头指定数量的标签片段。',
    recommendation: '按你放在开头的固定标签数量设置。没有需要保护的标签时可以设为 0。',
    note: '数量按标注分隔符（默认逗号）拆开的片段计数，不是文本编码器的分词数。整条标注丢弃不受这一项保护。',
  },
  prefer_json_caption: {
    summary: '兼容旧配置的标注开关：未指定来源模式时优先读取同名 JSON，再回退文本。',
    recommendation: '新蓝图直接使用“标注来源”选择格式。不要依靠这个旧开关改变显式来源模式。',
    note: '已经选择 auto、txt、json 或 captions_json 时，本开关不会改变该模式的读取规则。',
  },
  caption_source_mode: {
    summary: '选择预处理、缓存和图片预览读取哪种文字标注。',
    recommendation: '混合格式先用“自动识别”，它依次尝试 captions.json → 同名 JSON → 同名文本。必须只读一种格式时，再选择对应来源。',
    options: [
      { value: 'auto', label: '自动识别', when: '数据有多种格式，希望按顺序寻找标注。' },
      { value: 'txt', label: '同名文本', when: '只读与图片同名的文本，后缀由“标注扩展名”指定。' },
      { value: 'json', label: '同名 JSON', when: '只读每张图片对应的结构化 JSON。' },
      { value: 'captions_json', label: 'captions.json', when: '只读目录中记录图片与标注对应关系的 captions.json。' },
    ],
    note: '强制指定一种格式后，不会自动尝试其他来源。改完标注来源或内容后，需要重新生成文本缓存。',
    advanced: ['没有文本缓存、直接从训练图片目录读取标注的路径，只读取“标注扩展名”对应的文本。不要把预览支持 JSON 理解为所有训练读取路径都直接支持 JSON。'],
  },
  source_dir: {
    summary: '存放原始图片和文字标注的目录，预处理会从这里取数据。',
    recommendation: '指向你整理好的原图目录。相对路径从项目根目录计算，也可以直接填写绝对路径。',
    note: '先确认路径检测到的图片数量，再用图片工作台检查标注。Qwen 编辑前子集在这里提供参考图片。',
  },
  image_dir: {
    summary: '存放缩放、裁剪后的训练图片的目录，区别于原始图片目录。',
    recommendation: '先沿用当前蓝图的处理目录。需要管理预处理产物时再改，并确认该目录对应同一份数据。',
    note: 'WebUI 启动训练时可能把处理产物绑定到运行时缓存目录，实际路径以本次训练配置为准。Qwen 编辑后子集在这里读取目标图。',
  },
  cache_dir: {
    summary: '存放图像和文字编码缓存，避免每次训练都重新计算。',
    recommendation: '你想让缓存与图片分开放时，再指定独立目录。相对路径从项目根目录计算。',
    note: '更换模型、图片或标注后，旧缓存可能需要重建，不能只靠填写同一目录保证兼容。',
    advanced: ['缓存包含 latent（压缩后的图像特征）和文本编码等文件。WebUI 的实际缓存位置还受运行时缓存池绑定影响。'],
  },
  num_repeats: {
    summary: '每轮训练把这个子集的图片重复多少次，例如 3 表示每张图参与三次。',
    recommendation: '先用 1，只有需要提高这个子集在训练中的占比时再增加。增加后请重新查看训练步数估算。',
    note: '它只影响训练集，验证集的重复次数固定为 1。',
  },
  recursive: {
    summary: '决定是否同时扫描图片目录中的子文件夹。',
    recommendation: '图片分散在子文件夹时开启。只想读取当前这一层目录时关闭。',
    note: '开启后仍会应用“路径筛选”，不会忽略筛选规则。',
  },
  path_pattern: {
    summary: '按图片的相对路径筛选，例如 portraits/* 只保留对应子目录中的图片。',
    recommendation: '不筛选时保持 *。多个规则用 | 连接，例如 portraits/*|landscapes/*，满足其中一个即可。',
    note: '读取子目录还需要开启“递归扫描子目录”。填错规则可能筛掉全部图片，请核对预览数量。',
    advanced: ['匹配使用 fnmatch（支持 * 和 ? 的路径通配规则），不是正则表达式。匹配对象是图片相对所扫描目录的路径。'],
  },
  is_reg: {
    summary: '把这个子集作为正则化图片，用来约束模型对原有类别的学习。',
    recommendation: '只有你明确准备了正则图片时才开启，并检查“正则损失权重”。普通训练图片保持关闭。',
    note: '正则数据和 Qwen 编辑数据都不能用于分阶段调度。',
    advanced: ['正则图片的重复注册会按普通图片的训练量配平，可能被普通数据的轮数截断。不能只用这里的重复次数推算最终训练量。'],
  },
  edit_role: {
    summary: '为 Qwen 图像编辑指定参考图和目标图，普通文生图数据使用“非编辑”。',
    recommendation: '编辑任务用一个“编辑前”和一个“编辑后”子集配成一对。两侧填写相同配对编号。',
    options: [
      { value: 'normal', label: '非编辑', when: '普通图片训练，可按需要另行标记正则数据。' },
      { value: 'before', label: '编辑前', when: '提供参考图，不作为普通训练样本。' },
      { value: 'after', label: '编辑后', when: '提供目标图和编辑指令，批大小必须为 1。' },
    ],
    note: '同一份编辑蓝图不能混入普通或正则子集，也不能启用分阶段调度。',
  },
  edit_pair_id: {
    summary: '用同一个编号或名称，把编辑前参考子集和编辑后目标子集配在一起。',
    recommendation: '每对使用一个唯一编号，并确保两侧各出现一次。对应图片保持相同子目录和文件主名（不含扩展名）。',
    note: '保存时检查每个编号是否恰好有一前一后。应用到训练配置及开训时还会检查图片配对，缺失或歧义会被拒绝。',
    advanced: ['图片按相对子目录和文件主名配对，大小写不敏感。两侧扩展名可以不同，但同一配对键不能对应多张图片。'],
  },
  mask_mode: {
    summary: '选择图片的哪些区域参与训练，白色区域参与损失，黑色区域被忽略。',
    recommendation: '整图训练选“不使用蒙版”，已有独立蒙版文件选“外部蒙版目录”。只有需要兼容旧目录时才选“兼容自动发现”。',
    options: [
      { value: 'none', label: '不使用蒙版', when: '整张图片都参与训练。' },
      { value: 'external', label: '外部蒙版目录', when: '读取独立蒙版文件，需要填写外部蒙版目录。' },
      { value: 'embedded', label: '图像 Alpha', when: '使用图片的透明度通道，透明区域不参与损失。' },
      { value: 'auto', label: '兼容自动发现', when: '尝试发现旧的 masks 目录或读取图片 Alpha。' },
    ],
    note: '缺失蒙版时回退整图权重，不会自动跳过图片。图像编辑训练使用蒙版的限制还需按模型预检结果检查。',
  },
  mask_dir: {
    summary: '指定外部蒙版的目录，例如 cat.jpg 对应 cat_mask.png。',
    recommendation: '“蒙版模式”选 external 时填写。按图片的子目录结构保存蒙版，方便你检查对应关系。',
    note: '找不到蒙版会按整图训练，请在开训前检查实际文件。切换到其他模式保存时会清除本项。',
    advanced: ['查找文件名为 {stem}_mask.png，先尝试对应子目录，再尝试蒙版目录根部的同名文件（flat fallback）。相对目录从项目根目录计算，也支持绝对路径。'],
  },
  nl_tag_mix_enabled: {
    summary: '调整 captions.json 中自然语言描述（NL）与标签描述（Tag）的训练比例。',
    recommendation: '每张图片已准备这两类文字变体时再开启。它不会自动把自然语言改写为标签。',
    note: '只有一条标注的图片仍全部保留，不能靠本开关改变这类图片的最终占比。',
  },
  tag_ratio: {
    summary: '多标注混合时希望标签描述占多少，例如 0.7 表示约 70%。',
    recommendation: '先沿用已有比例，再根据两类标注的质量调整。调高偏向标签描述，调低偏向自然语言描述。',
    note: '只有一条标注的图片仍保留原文，比例不会生成另一种描述。缺少某类文字时，系统会用现有文字补足。',
    advanced: ['captions.json 的每图多条文字会按比例重复选取，保持变体总数。比例会取整，因此少量变体的实际占比不一定等于你填写的数值。'],
  },
  trigger_clone_enabled: {
    summary: '额外复制一份图片子集，把副本的整条标注替换为指定的触发文字。',
    recommendation: '只有你需要额外的“图片 + 单一触发文字”训练样本时再开启。先用少量重复次数检查训练占比。',
    note: '原子集仍保留，副本独立计算重复次数，会增加训练量和预处理产物。它不会在原标注前简单追加触发词。',
  },
  trigger_clone_prompt: {
    summary: '副本样本使用的完整文字标注，例如 my_character_token。',
    recommendation: '填写你希望与这组图片关联的词或短语。开启复制后不能为空。',
    note: '副本只使用这段文字，原标注不会保留在副本中。',
  },
  trigger_clone_num_repeats: {
    summary: '额外复制子集每轮参与训练的次数，独立于原子集的“重复次数”。',
    recommendation: '先用 1，再根据原子集和副本的训练占比调整。调高会增加副本样本的影响和总训练量。',
  },
  stage_schedule_enabled: {
    summary: '按训练进度分段，让每个阶段使用你指定的数据子集。',
    recommendation: '普通训练先保持关闭。需要先学习一组数据、再强化另一组时开启，并用“配置阶段”设置顺序和时长。',
    note: '全部阶段必须连续覆盖 0% 到 100%，不能混用正则化或 Qwen 图像编辑数据。',
  },
  stage_name: {
    summary: '阶段在时间线中的显示名称，方便你分辨各段训练。',
    recommendation: '使用能说明目的的短名称，例如“整体学习”和“细节强化”。名称本身不会改变学习率或训练行为。',
  },
  stage_subset_index: {
    summary: '选择这一段训练使用哪个数据子集。',
    recommendation: '按子集目录和编号选择，确认选中的是你希望在该阶段学习的数据。',
    note: '正则化子集和 Qwen 图像编辑蓝图暂不支持分阶段调度。',
  },
  stage_start_pct: {
    summary: '这一阶段从训练总进度的哪个位置开始，例如 30 表示 30%。',
    recommendation: '第一阶段从 0% 开始，其余阶段紧接上一阶段结束。界面会联动相邻边界，避免留出空档。',
    note: '每一段都必须有正长度，全部阶段连续覆盖 0% 到 100%。',
  },
  stage_end_pct: {
    summary: '这一阶段到训练总进度的哪个位置结束。',
    recommendation: '最后一阶段结束于 100%，其余阶段按你希望分配的训练量调整。移动阶段只交换名称和子集绑定，不改变时间区间。',
    note: '百分比表示总训练进度，不是这个子集的图片占比或重复次数。',
  },
} satisfies Record<string, DatasetHelpContent>;

export type DatasetFieldHelpKey = keyof typeof DATASET_FIELD_HELP;
