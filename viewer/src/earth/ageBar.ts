/**
 * The age slider along the bottom: the run's saved ages, oldest on the left, the present on
 * the right. The thumb moves by index through the saved steps, so every age it reports has
 * data. Ages are geological (Ma): these runs follow a real plate reconstruction.
 */
export class AgeBar {
  private readonly slider: HTMLInputElement;
  private readonly value: HTMLSpanElement;
  private ages: number[] = [0];

  constructor(private readonly onChange: (age: number) => void) {
    const bar = document.createElement('div');
    bar.className = 'asth-box asth-agebar';
    const label = document.createElement('span');
    label.textContent = 'Age';
    this.value = document.createElement('span');
    this.value.className = 'asth-agevalue';
    this.slider = document.createElement('input');
    this.slider.type = 'range';
    this.slider.className = 'asth-scrub';
    this.slider.min = '0';
    this.slider.step = '1';
    this.slider.addEventListener('input', () => {
      this.paint();
      this.onChange(this.age);
    });
    bar.append(label, this.value, this.slider);
    document.body.appendChild(bar);
  }

  /** Saved ages in any order; shown oldest first. */
  setAges(ages: number[]): void {
    const keep = this.age;
    this.ages = [...ages].sort((a, b) => b - a);
    this.slider.max = String(this.ages.length - 1);
    this.setAge(keep);
  }

  get age(): number { return this.ages[Number(this.slider.value)] ?? this.ages[0]; }

  /** Move to the saved age nearest `age`. */
  setAge(age: number): void {
    let best = 0;
    this.ages.forEach((a, i) => { if (Math.abs(a - age) < Math.abs(this.ages[best] - age)) best = i; });
    this.slider.value = String(best);
    this.paint();
  }

  private paint(): void { this.value.textContent = `${this.age} Ma`; }
}
